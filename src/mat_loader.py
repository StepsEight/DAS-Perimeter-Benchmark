"""Read MATLAB DAS patches without modifying or copying source MAT files.

Selection is exact-shape only: a variable must contain (2000, 50) patches,
possibly transposed, or a stack of those patches. Preferred variable names
break ties before lexical order. MATLAB v7.3 storage reverses MATLAB axes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat, whosmat


PREFERRED_VARIABLES = ("instances", "img", "data", "X", "x")


@dataclass
class MatInspection:
    path: Path
    kind: str
    variables: list[dict[str, Any]]
    selected_variable: str
    original_shape: tuple[int, ...]
    storage_shape: tuple[int, ...]
    layout: str
    n_instances: int
    dtype: str
    metadata: dict[str, Any]
    decision: str


def natural_key(value: str) -> list[Any]:
    return [int(v) if v.isdigit() else v.lower() for v in re.split(r"(\d+)", value)]


def _text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.dtype.kind in "ui":
        return "".join(chr(int(v)) for v in arr.ravel(order="F") if v)
    if arr.dtype.kind in "US":
        return "".join(str(v) for v in arr.ravel()).strip()
    return str(value)


def _h5_value(handle: h5py.File, item: h5py.Dataset) -> Any:
    value = item[()]
    matlab_class = item.attrs.get("MATLAB_class", b"")
    if matlab_class == b"char":
        return _text(value)
    if h5py.check_dtype(ref=item.dtype) is not None:
        return [_h5_value(handle, handle[ref]) if ref else None
                for ref in np.asarray(value).ravel(order="C")]
    return value


def _layout(shape: tuple[int, ...], expected: tuple[int, int], hdf5: bool) -> tuple[str, int] | None:
    t, c = expected
    if len(shape) == 2:
        if shape == (t, c):
            return "TC", 1
        if shape == (c, t):
            return "CT", 1
    if len(shape) == 3:
        # Native MATLAB aggregate: T,C,N. v7.3 exposes N,C,T to h5py.
        choices = [("NCT", shape[0])] if shape[1:] == (c, t) else []
        if shape[:2] == (t, c):
            choices.append(("TCN", shape[2]))
        if shape[1:] == (t, c):
            choices.append(("NTC", shape[0]))
        if choices:
            if hdf5:
                choices.sort(key=lambda q: q[0] != "NCT")
            else:
                choices.sort(key=lambda q: q[0] != "TCN")
            return choices[0]
    return None


def inspect_mat(path: str | Path, expected_shape: tuple[int, int] = (2000, 50)) -> MatInspection:
    path = Path(path).resolve()
    is_hdf5 = h5py.is_hdf5(path)
    variables: list[dict[str, Any]] = []
    candidates: list[tuple[str, tuple[int, ...], str, str, int]] = []
    metadata: dict[str, Any] = {}
    if is_hdf5:
        with h5py.File(path, "r") as handle:
            for name, item in handle.items():
                if name.startswith("#"):
                    continue
                if not isinstance(item, h5py.Dataset):
                    variables.append({"name": name, "shape": None, "dtype": "struct"})
                    continue
                shape = tuple(item.shape)
                variables.append({"name": name, "storage_shape": list(shape), "dtype": str(item.dtype)})
                layout = _layout(shape, expected_shape, True)
                if item.dtype.kind in "fiu" and layout and item.attrs.get("MATLAB_class") != b"char":
                    candidates.append((name, shape, str(item.dtype), *layout))
                elif item.size < 100_000:
                    metadata[name] = _h5_value(handle, item)
    else:
        for name, shape, kind in whosmat(path):
            shape = tuple(shape)
            variables.append({"name": name, "shape": list(shape), "dtype": kind})
            layout = _layout(shape, expected_shape, False)
            if kind in {"double", "single", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"} and layout:
                candidates.append((name, shape, kind, *layout))
        if candidates:
            candidate_names = {v[0] for v in candidates}
            small_names = [v["name"] for v in variables if v["name"] not in candidate_names]
            small = loadmat(path, variable_names=small_names, squeeze_me=True)
            metadata = {k: (_text(v) if np.asarray(v).dtype.kind in "US" else v)
                        for k, v in small.items() if not k.startswith("__")}
    if not candidates:
        raise ValueError(f"{path}: no numerical variable with exact patch shape {expected_shape}; variables={variables}")
    ranks = {name: idx for idx, name in enumerate(PREFERRED_VARIABLES)}
    candidates.sort(key=lambda v: (ranks.get(v[0], len(ranks)), v[0]))
    name, storage, dtype, layout, count = candidates[0]
    original = tuple(reversed(storage)) if is_hdf5 else storage
    decision = (f"Selected {name}; exact patch-shape candidates={[v[0] for v in candidates]}; "
                "tie-break preferred names instances,img,data,X,x then lexical; no reshape.")
    return MatInspection(path, "v7.3-hdf5" if is_hdf5 else "legacy-mat", variables,
                         name, original, storage, layout, count, dtype, metadata, decision)


def metadata_scalar(metadata: dict[str, Any], name: str, default: Any = None) -> Any:
    if name not in metadata:
        return default
    value = metadata[name]
    if isinstance(value, str):
        return value
    values = np.asarray(value).ravel()
    return values[0].item() if values.size == 1 else value


def metadata_vector(metadata: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(metadata.get(name, [])).ravel()


def read_samples(inspection: MatInspection, instance_indices: list[int] | np.ndarray | None = None,
                 restore_amplitude: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return [N,T,C] float32 patches, valid-time mask, and explicit warnings.

    ``instance_indices`` uses zero-based indices. Amplitude restoration applies
    only on valid rows; padding remains zero instead of becoming a physical DC.
    """
    indices = np.arange(inspection.n_instances) if instance_indices is None else np.asarray(instance_indices, dtype=int)
    if np.any(indices < 0) or np.any(indices >= inspection.n_instances):
        raise IndexError("MAT instance index outside available range")
    if inspection.kind == "v7.3-hdf5":
        with h5py.File(inspection.path, "r") as handle:
            item = handle[inspection.selected_variable]
            if inspection.layout == "NCT":
                # Keep HDF5 reads contiguous when selecting first N samples.
                if len(indices) and np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices))):
                    data = item[int(indices[0]):int(indices[-1]) + 1].transpose(0, 2, 1)
                else:
                    data = np.stack([item[int(i)].T for i in indices])
            elif inspection.layout == "TCN":
                data = np.stack([item[:, :, int(i)] for i in indices])
            elif inspection.layout == "NTC":
                data = np.stack([item[int(i)] for i in indices])
            else:
                data = item[()]
                data = (data if inspection.layout == "TC" else data.T)[None]
    else:
        data = loadmat(inspection.path, variable_names=[inspection.selected_variable])[inspection.selected_variable]
        if inspection.layout == "TCN":
            data = data[:, :, indices].transpose(2, 0, 1)
        elif inspection.layout == "NCT":
            data = data[indices].transpose(0, 2, 1)
        elif inspection.layout == "NTC":
            data = data[indices]
        else:
            data = (data if inspection.layout == "TC" else data.T)[None]
    data = np.asarray(data, dtype=np.float32)
    valid = metadata_vector(inspection.metadata, "valid_data_mask").astype(bool)
    if not valid.size:
        valid = np.ones(data.shape[1], dtype=bool)
    if valid.shape != (data.shape[1],) or not valid.any():
        raise ValueError(f"{inspection.path}: invalid valid_data_mask")
    warnings: list[str] = []
    if restore_amplitude:
        lo = metadata_vector(inspection.metadata, "normalization_min")
        hi = metadata_vector(inspection.metadata, "normalization_max")
        if lo.size >= inspection.n_instances and hi.size >= inspection.n_instances:
            scale = (hi[indices] - lo[indices]).astype(np.float64)
            if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError(f"{inspection.path}: invalid normalization range")
            # Per-patch operations avoid another full double-precision data copy.
            for n, original_index in enumerate(indices):
                restored = data[n, valid, :].astype(np.float64) * scale[n] + lo[original_index]
                data[n, valid, :] = restored.astype(np.float32)
            data[:, ~valid, :] = 0
        else:
            warnings.append("Amplitude restoration requested but min/max metadata missing; retaining stored values.")
    return data, valid, warnings


def discover_sources(data_root: str | Path, expected_shape: tuple[int, int] = (2000, 50)) -> tuple[dict[str, list[MatInspection]], list[dict[str, str]]]:
    """Discover aggregate MAT labels from event_name, or legacy class folders."""
    root = Path(data_root).resolve()
    grouped: dict[str, list[MatInspection]] = {}
    rejected: list[dict[str, str]] = []
    paths = list(root.glob("*.mat"))
    paths += [p for directory in root.iterdir() if directory.is_dir() and not directory.name.startswith(".")
              for p in directory.glob("*.mat")]
    for path in sorted(paths, key=lambda p: natural_key(str(p))):
        try:
            inspection = inspect_mat(path, expected_shape)
            event = metadata_scalar(inspection.metadata, "event_name")
            if not event:
                event = path.parent.name if path.parent != root else path.stem
            grouped.setdefault(str(event), []).append(inspection)
        except (OSError, ValueError, TypeError) as exc:
            rejected.append({"file_path": str(path), "reason": str(exc)})
    return grouped, rejected

"""Physical amplitude restoration lives in mat_loader; DL normalization here."""
from __future__ import annotations

import numpy as np
from scipy import signal


def standardize_waveform(waveform: np.ndarray, valid_mask: np.ndarray | None = None,
                         eps: float = 1e-8) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float64)
    valid = np.ones(waveform.shape[-1], bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if waveform.ndim == 1:
        output = np.zeros_like(waveform)
        values = waveform[valid]
        output[valid] = (values - values.mean()) / (values.std() + eps)
    elif waveform.ndim == 2:
        output = np.zeros_like(waveform)
        values = waveform[:, valid]
        output[:, valid] = (values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + eps)
    else:
        raise ValueError("waveform must be [T] or [N,T]")
    return output.astype(np.float32)


def standardize_patch(patch: np.ndarray, valid_mask: np.ndarray | None = None,
                      eps: float = 1e-8) -> np.ndarray:
    """One mean/std over valid time x all 50 channels; return [space,time]."""
    patch = np.asarray(patch, dtype=np.float64)
    if patch.ndim != 2:
        raise ValueError("patch must have time x channel dimensions")
    valid = np.ones(patch.shape[0], bool) if valid_mask is None else np.asarray(valid_mask, bool)
    values = patch[valid, :]
    output = np.zeros_like(patch)
    output[valid, :] = (values - values.mean()) / (values.std() + eps)
    return output.T.astype(np.float32)


def compute_stft(waveform: np.ndarray, fs: float = 2000, nperseg: int = 256,
                 noverlap: int = 192, nfft: int = 256, eps: float = 1e-12,
                 normalize: bool = True, boundary: str = "zeros", padded: bool = True,
                 window: str = "hann") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-sided log-power STFT, natural logarithm, sample-wise normalization.

    With the documented 2000-point input and 256/192/256 parameters, SciPy's
    zero boundary extension and padded final frame produce exactly 129 x 33.
    """
    frequencies, times, z = signal.stft(np.asarray(waveform, dtype=np.float64), fs=fs,
                                       window=window, nperseg=nperseg, noverlap=noverlap,
                                       nfft=nfft, boundary=boundary, padded=padded,
                                       return_onesided=True, detrend=False)
    log_power = np.log(np.abs(z) ** 2 + eps)
    if normalize:
        log_power = (log_power - log_power.mean()) / (log_power.std() + 1e-8)
    return frequencies, times, log_power.astype(np.float32)

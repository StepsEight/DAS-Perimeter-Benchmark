"""Portable readers for the four aligned paper representations and fixed splits."""
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd

try:
    from .features import FEATURE_NAMES
    from .mat_loader import inspect_mat, read_samples
    from .preprocessing import standardize_patch
    from .utils import ROOT, load_config, sha256_file
except ImportError:
    from features import FEATURE_NAMES
    from mat_loader import inspect_mat, read_samples
    from preprocessing import standardize_patch
    from utils import ROOT, load_config, sha256_file

CLASSES = ['background', 'digging', 'raining', 'vehicle', 'walking']
_HASH_CACHE = {}
_ARRAY_CACHE = {}


def data_directory(config=None):
    config = config or load_config()
    directory = Path(config.get('paths', {}).get('data_dir', 'data'))
    return directory if directory.is_absolute() else ROOT / directory


def verify_paths(paths, config=None):
    """Check release hashes once per file version; refuse missing/changed inputs."""
    manifest = json.loads((data_directory(config) / 'manifest.json').read_text(encoding='utf-8'))
    expected = {record['path']: record for record in manifest['files']}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f'{relative} is missing. Run python scripts/download_data.py')
        if relative not in expected:
            raise ValueError(f'Unregistered data file: {relative}')
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in _HASH_CACHE:
            record = expected[relative]
            if stat.st_size != record['bytes'] or sha256_file(path) != record['sha256']:
                raise ValueError(f'Release checksum mismatch: {relative}')
            _HASH_CACHE[key] = record['sha256']
    return manifest


def validate_metadata(frame, split_frames):
    """The official protocol is fixed at 4000 ordered samples and 1000 BIN groups."""
    if len(frame) != 4000 or not frame.sample_id.is_unique:
        raise ValueError('Expected 4000 unique sample IDs')
    if not np.array_equal(frame.row_index.to_numpy(), np.arange(4000)):
        raise ValueError('Metadata row order changed')
    if list(frame.class_name.drop_duplicates()) != CLASSES:
        raise ValueError('Class order changed')
    if not np.array_equal(frame.class_index.to_numpy(), np.repeat(np.arange(5), 800)):
        raise ValueError('Class labels changed')
    for name, rows in frame.groupby('class_name', sort=False):
        if not np.array_equal(rows.instance_index.to_numpy(), np.arange(1, 801)):
            raise ValueError(f'Chronological instance order changed: {name}')
        if rows.sample_id.tolist() != [f'{name}_{i:06d}' for i in range(1, 801)]:
            raise ValueError(f'Sample ID mapping changed: {name}')
        if rows.split.tolist() != ['train']*480 + ['val']*160 + ['test']*160:
            raise ValueError(f'Chronological split changed: {name}')
        if not np.array_equal(rows.source_bin_index.to_numpy(), np.repeat(np.arange(1, 201), 4)):
            raise ValueError(f'Source BIN order changed: {name}')
    if not frame.groupby('group_id').size().eq(4).all() or frame.group_id.nunique() != 1000:
        raise ValueError('Each of the 1000 recordings must have four spatial patches')
    for key in ['group_id', 'bin_sha256']:
        if frame.groupby(key).split.nunique().max() != 1:
            raise ValueError(f'Source recording leakage across splits: {key}')
    for split in ['train', 'val', 'test']:
        expected = frame.loc[frame.split.eq(split), ['sample_id', 'row_index']].reset_index(drop=True)
        actual = split_frames[split][['sample_id', 'row_index']].reset_index(drop=True)
        if not expected.equals(actual):
            raise ValueError(f'Fixed membership/order mismatch: {split}')
    return frame


def load_metadata(config=None):
    directory = data_directory(config)
    paths = ['data/metadata.csv', 'data/class_mapping.json',
             *[f'data/splits/{s}.csv' for s in ['train', 'val', 'test']]]
    verify_paths(paths, config)
    frame = pd.read_csv(directory / 'metadata.csv', float_precision='round_trip')
    splits = {s: pd.read_csv(directory / f'splits/{s}.csv') for s in ['train', 'val', 'test']}
    return validate_metadata(frame, splits)


def load_representation(method, config=None):
    """Return metadata and arrays in [sample,...] order, without a feature-plane axis."""
    config = config or load_config()
    index = load_metadata(config)
    directory = data_directory(config)
    files = {
        'svm': ['data/handcrafted_features.csv'],
        'cnn1d': ['data/temporal/DAS_1D_2kHz_4000.mat'],
        'stft_cnn': ['data/time_frequency/stft.npz'],
        'cnn2d': [f'data/spatiotemporal/{name}.mat' for name in CLASSES],
    }
    if method not in files:
        raise ValueError(f'Unknown paper baseline: {method}')
    verify_paths(files[method], config)
    key = (method, tuple((p, (ROOT/p).stat().st_mtime_ns) for p in files[method]))
    if key in _ARRAY_CACHE:
        return index, _ARRAY_CACHE[key]
    if method == 'svm':
        features = pd.read_csv(directory/'handcrafted_features.csv', float_precision='round_trip')
        if features.sample_id.tolist() != index.sample_id.tolist():
            raise ValueError('Feature IDs do not match metadata')
        values = features[FEATURE_NAMES].to_numpy(np.float64)
    elif method == 'cnn1d':
        with h5py.File(directory/'temporal/DAS_1D_2kHz_4000.mat', 'r') as handle:
            values = handle['waveforms'][:].T.copy()
            if not np.array_equal(handle['labels'][:].ravel(), index.class_index.to_numpy()):
                raise ValueError('Temporal labels do not match metadata')
    elif method == 'stft_cnn':
        with np.load(directory/'time_frequency/stft.npz', allow_pickle=False) as archive:
            values = archive['stft']
            if not np.array_equal(archive['sample_ids'], index.sample_id.to_numpy()):
                raise ValueError('STFT IDs do not match metadata')
            if not np.array_equal(archive['labels'], index.class_index.to_numpy()):
                raise ValueError('STFT labels do not match metadata')
    else:
        values = np.empty((4000, 50, 2000), dtype=np.float32)
        for name, rows in index.groupby('class_name', sort=False):
            inspection = inspect_mat(directory / f'spatiotemporal/{name}.mat')
            patches, valid, warnings = read_samples(inspection, np.arange(800), restore_amplitude=True)
            if warnings:
                raise ValueError('; '.join(warnings))
            for local, position in enumerate(rows.row_index):
                # Preserve the original baseline's float32 amplitude-restoration
                # round trip before float64 mean/std calculation.
                values[position] = standardize_patch(patches[local], valid)
            print(f'Loaded {name}: 800 space-time samples', flush=True)
    shapes = {'svm': (4000, 15), 'cnn1d': (4000, 2000),
              'cnn2d': (4000, 50, 2000), 'stft_cnn': (4000, 129, 33)}
    if values.shape != shapes[method] or not np.isfinite(values).all():
        raise ValueError(f'Invalid {method} representation')
    _ARRAY_CACHE[key] = values
    return index, values


def get_feature_data(config=None):
    index, features = load_representation('svm', config)
    result = {}
    for split in ['train', 'val', 'test']:
        mask = index.split.eq(split).to_numpy()
        result[f'X_{split}'] = features[mask]
        result[f'y_{split}'] = index.class_index.to_numpy(np.int64)[mask]
        result[f'ids_{split}'] = index.sample_id.to_numpy(str)[mask]
    return result


class RepresentationDataset:
    def __init__(self, inputs, labels, sample_ids, positions):
        self.inputs, self.labels, self.sample_ids, self.positions = inputs, labels, sample_ids, positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, item):
        position = int(self.positions[item])
        return self.inputs[position][None], int(self.labels[position]), str(self.sample_ids[position])


def get_dataloaders(config=None, method='cnn1d'):
    import torch
    from torch.utils.data import DataLoader
    config = config or load_config()
    index, inputs = load_representation(method, config)
    training = config['training']
    generator = torch.Generator().manual_seed(int(config['seed']))
    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = RepresentationDataset(inputs, index.class_index.to_numpy(np.int64),
            index.sample_id.to_numpy(str), np.flatnonzero(index.split.eq(split).to_numpy()))
        loaders[split] = DataLoader(dataset, batch_size=int(training['batch_size']),
            shuffle=split=='train', num_workers=int(training['num_workers']),
            pin_memory=bool(training['pin_memory']), drop_last=False,
            generator=generator if split=='train' else None)
    return loaders

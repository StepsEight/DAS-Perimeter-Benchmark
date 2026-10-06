"""Regenerate the single-channel paper representations from the five released MATs."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import h5py
import numpy as np
import pandas as pd
from dataset import load_metadata, load_representation, CLASSES, verify_paths
from features import extract_features, FEATURE_NAMES
from preprocessing import standardize_waveform, compute_stft
from utils import ROOT


def regenerate():
    index = load_metadata()
    verify_paths([f'data/spatiotemporal/{name}.mat' for name in CLASSES])
    waveforms = np.empty((4000, 2000), np.float32)
    stft = np.empty((4000, 129, 33), np.float32)
    features = np.empty((4000, 15), np.float64)
    for name, rows in index.groupby('class_name', sort=False):
        with h5py.File(ROOT/f'data/spatiotemporal/{name}.mat', 'r') as handle:
            center = handle['instances'][:, 24, :].astype(np.float64)
            lo, hi = [handle[field][:].ravel() for field in ['normalization_min','normalization_max']]
            valid = handle['valid_data_mask'][:].ravel().astype(bool)
        for local, position in enumerate(rows.row_index):
            raw = center[local]
            raw[valid] = raw[valid] * (hi[local]-lo[local]) + lo[local]
            raw[~valid] = 0
            waveforms[position] = standardize_waveform(raw, valid)
            features[position] = extract_features(raw)
            frequencies, times, stft[position] = compute_stft(waveforms[position])
        print(f'Regenerated {name}: 800 aligned instances', flush=True)
    return index, waveforms, features, stft, frequencies, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', help='Compare to released arrays (default)')
    group.add_argument('--output', type=Path, help='Write regenerated NPZ/CSV files to an empty directory')
    args = parser.parse_args()
    if args.output and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Regeneration output must be empty')
    index, waveforms, features, stft, frequencies, times = regenerate()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output/'waveforms.npz', waveforms=waveforms,
                            sample_ids=index.sample_id.to_numpy(str),labels=index.class_index.to_numpy())
        np.savez_compressed(args.output/'stft.npz', stft=stft, frequencies_hz=frequencies,
                            times_seconds=times, sample_ids=index.sample_id.to_numpy(str),
                            labels=index.class_index.to_numpy())
        frame=index[['sample_id','class_index','split']].copy()
        for j,name in enumerate(FEATURE_NAMES): frame[name]=features[:,j]
        frame.to_csv(args.output/'handcrafted_features.csv',index=False)
    else:
        for method, actual in [('cnn1d',waveforms),('svm',features),('stft_cnn',stft)]:
            _,expected=load_representation(method)
            difference=float(np.max(np.abs(actual-expected)))
            # Float32 representations must match exactly in the recorded environment.
            # Float64 features permit only reduction-order roundoff.
            if method=='svm':
                np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
            else:
                np.testing.assert_array_equal(actual,expected)
            print(f'{method}: matches release; maximum absolute difference={difference:.3g}')


if __name__=='__main__':
    main()

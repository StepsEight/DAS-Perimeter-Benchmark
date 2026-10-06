"""Read the same sample in all four representations using only data dependencies."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dataset import load_metadata,load_representation
from mat_loader import inspect_mat,read_samples
from preprocessing import standardize_patch
from utils import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-id',default='walking_000241')
    args=parser.parse_args()
    metadata=load_metadata()
    rows=metadata[metadata.sample_id.eq(args.sample_id)]
    if len(rows)!=1: raise ValueError('Unknown sample ID')
    row=rows.iloc[0]
    print(f'{row.sample_id}: class={row.class_name}, split={row.split}, recording group={row.group_id}')
    for method in ['svm','cnn1d','stft_cnn']:
        _,values=load_representation(method)
        print(f'{method}: shape={values[int(row.row_index)].shape}, dtype={values.dtype}')
    source=inspect_mat(ROOT/row.spatiotemporal_file)
    raw,valid,warnings=read_samples(source,[int(row.instance_index)-1],restore_amplitude=True)
    if warnings: raise ValueError(warnings)
    patch=standardize_patch(raw[0],valid)
    print(f'cnn2d: shape={patch.shape}, dtype={patch.dtype}; axes=space,time')


if __name__=='__main__':
    main()

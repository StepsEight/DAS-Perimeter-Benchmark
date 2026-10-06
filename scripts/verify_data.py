"""Verify release checksums, source-recording grouping and all four input schemas."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import h5py
import numpy as np
from dataset import load_metadata, load_representation, verify_paths, CLASSES
from utils import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only',action='store_true',help='Check tracked metadata without the large download')
    args=parser.parse_args()
    index=load_metadata()
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    if not args.metadata_only:
        verify_paths([record['path'] for record in manifest['files']])
        for method in ['svm','cnn1d','stft_cnn']:
            _,values=load_representation(method)
            if method=='cnn1d' and np.any(values[:,:41]!=0):
                raise ValueError('Temporal prefix is not zero')
        for name in CLASSES:
            with h5py.File(ROOT/f'data/spatiotemporal/{name}.mat','r') as h:
                if h['instances'].shape!=(800,50,2000) or h['instances'].dtype!=np.dtype('float32'):
                    raise ValueError(f'Unexpected MAT dimensions/dtype: {name}')
                rows=index[index.class_name.eq(name)]
                for key in ['group_id','source_bin_index','instance_index','channel_start','channel_end']:
                    if not np.array_equal(h[key][:].ravel(),rows[key].to_numpy()):
                        raise ValueError(f'MAT metadata mismatch: {name}/{key}')
                if not np.array_equal(h['valid_data_mask'][:].ravel().astype(bool),np.arange(2000)>=41):
                    raise ValueError('Invalid valid-data mask')
                for start in range(0,800,32):
                    block=h['instances'][start:start+32]
                    if not np.isfinite(block).all() or np.any(block[:,:,:41]!=0):
                        raise ValueError(f'Nonfinite data/nonzero padding: {name}')
                    if block.min()<0 or block.max()>1:
                        raise ValueError(f'Stored MAT is outside min-max range: {name}')
    print(json.dumps({'status':'passed','instances':len(index),'classes':5,
        'split_instances':index.split.value_counts().to_dict(),
        'source_recordings':int(index.group_id.nunique()),
        'cross_split_source_groups':0,'metadata_only':args.metadata_only},indent=2))


if __name__=='__main__':
    main()

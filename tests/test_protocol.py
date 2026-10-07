"""Small protocol checks; no training and no dataset download required."""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dataset import validate_metadata
from preprocessing import standardize_waveform,standardize_patch,compute_stft
from features import extract_features


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.index=pd.read_csv(ROOT/'data/metadata.csv')
        self.splits={s:pd.read_csv(ROOT/f'data/splits/{s}.csv') for s in ['train','val','test']}

    def test_fixed_membership_and_grouping(self):
        validate_metadata(self.index,self.splits)
        self.assertEqual(self.index.groupby('split').group_id.nunique().to_dict(),
                         {'test':200,'train':600,'val':200})

    def test_cross_split_group_rejected(self):
        altered=self.index.copy()
        a,b=altered.loc[0,'group_id'],altered.loc[480,'group_id']
        altered.loc[0,'group_id']=b
        altered.loc[480,'group_id']=a
        with self.assertRaisesRegex(ValueError,'leakage'):
            validate_metadata(altered,self.splits)

    def test_split_file_order_rejected(self):
        self.splits['val']=self.splits['val'].iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError,'membership/order'):
            validate_metadata(self.index,self.splits)

    def test_normalization_excludes_padding(self):
        valid=np.arange(2000)>=41
        wave=np.concatenate([np.zeros(41),np.linspace(-3,9,1959)])
        z=standardize_waveform(wave,valid)
        np.testing.assert_array_equal(z[:41],0)
        self.assertAlmostEqual(float(z[valid].mean()),0,places=6)
        self.assertAlmostEqual(float(z[valid].std()),1,places=6)
        patch=np.column_stack([wave*(i+1) for i in range(50)])
        result=standardize_patch(patch,valid)
        self.assertEqual(result.shape,(50,2000))
        np.testing.assert_array_equal(result[:,:41],0)
        self.assertGreater(result[-1,41:].std(),result[0,41:].std())

    def test_stft_schema_and_feature_energy(self):
        wave=np.sin(2*np.pi*100*np.arange(2000)/2000)
        f,t,z=compute_stft(wave)
        self.assertEqual(z.shape,(129,33))
        self.assertEqual((f[0],f[-1]),(0,1000))
        self.assertAlmostEqual(t[-1],1.024)
        self.assertAlmostEqual(float(z.mean()),0,places=6)
        features=extract_features(wave)
        self.assertEqual(features.shape,(15,))
        self.assertAlmostEqual(features[-2:].sum(),1,places=12)

    def test_network_dimensions_and_counts(self):
        import torch
        from models import build_model,parameter_count
        torch.set_num_threads(2)
        for name,shape,count in [('cnn1d',(1,1,2000),36357),
                                 ('cnn2d',(1,1,50,2000),168069),
                                 ('stft_cnn',(1,1,129,33),93765)]:
            model=build_model(name,5,0.3).eval()
            self.assertEqual(parameter_count(model),count)
            with torch.inference_mode():
                self.assertEqual(tuple(model(torch.zeros(shape)).shape),(1,5))

    def test_archive_license_mapping_preserves_code_license(self):
        spec=importlib.util.spec_from_file_location('download_data',ROOT/'scripts/download_data.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);destination=root/'checkout';destination.mkdir()
            (destination/'LICENSE').write_text('MIT code license',encoding='utf-8')
            payload=b'CC BY-NC data license'
            archive=root/'data.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('LICENSE',payload)
            record={'path':'LICENSE-DATA','archive_path':'LICENSE','bytes':len(payload),
                    'sha256':hashlib.sha256(payload).hexdigest(),'in_archive':True}
            module.extract_verified(archive,{'files':[record]},destination)
            self.assertEqual((destination/'LICENSE').read_text(),'MIT code license')
            self.assertEqual((destination/'LICENSE-DATA').read_bytes(),payload)
            record['path']='../escape.txt'
            with self.assertRaisesRegex(ValueError,'escapes'):
                module.extract_verified(archive,{'files':[record]},destination)
            self.assertFalse((root/'escape.txt').exists())

    def test_archive_traversal_rejected(self):
        spec=importlib.util.spec_from_file_location('download_data',ROOT/'scripts/download_data.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.txt','payload')
            manifest={'files':[{'path':'../escape.txt','bytes':7,'sha256':'invalid','in_archive':True}]}
            with self.assertRaisesRegex(ValueError,'escapes'):
                module.extract_verified(archive,manifest,root/'destination')
            self.assertFalse((root/'escape.txt').exists())


if __name__=='__main__':unittest.main()

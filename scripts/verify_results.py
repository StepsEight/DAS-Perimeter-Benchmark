"""Recalculate paper metrics from saved predictions, without model inference."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from dataset import load_metadata,CLASSES
from utils import ROOT,sha256_file


def main():
    index=load_metadata()
    records=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'))
    summary=pd.read_csv(ROOT/'results/summary.csv')
    for method,record in records.items():
        for split,key in [('val','validation'),('test','test')]:
            frame=pd.read_csv(ROOT/f'results/predictions/{method}_{split}.csv')
            expected=index[index.split.eq(split)]
            assert frame.sample_id.tolist()==expected.sample_id.tolist()
            np.testing.assert_array_equal(frame.true_class_index,expected.class_index)
            assert frame.predicted_class_index.between(0,4).all()
            p,r,f,_=precision_recall_fscore_support(frame.true_class_index,frame.predicted_class_index,
                labels=np.arange(5),zero_division=0)
            actual={'accuracy':accuracy_score(frame.true_class_index,frame.predicted_class_index),
                'macro_precision':p.mean(),'macro_recall':r.mean(),'macro_f1':f.mean()}
            for metric,value in actual.items():
                np.testing.assert_allclose(value,record[key][metric],rtol=0,atol=1e-14)
                if split=='test':
                    np.testing.assert_allclose(value,summary.loc[summary.method.eq(method),'test_'+metric].iloc[0],rtol=0,atol=1e-14)
        if method!='svm':
            history=pd.read_csv(ROOT/f'results/history/{method}.csv')
            assert history.epoch.tolist()==list(range(1,101))
            best=history.loc[history.val_macro_f1.idxmax()]
            assert int(best.epoch)==record['best_epoch']
            np.testing.assert_allclose(best.val_macro_f1,record['validation']['macro_f1'],atol=1e-14,rtol=0)
        print(f'{method}: saved validation/test metrics and selection verified')
    checkpoints=json.loads((ROOT/'checkpoints/manifest.json').read_text(encoding='utf-8'))
    for record in checkpoints['files']:
        assert sha256_file(ROOT/record['path'])==record['sha256']
    print('All four paper baselines verified; no model test inference was run.')


if __name__=='__main__':
    main()

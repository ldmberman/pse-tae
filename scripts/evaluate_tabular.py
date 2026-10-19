"""Reload saved all-class logistic/XGBoost probes; never fit or select a model."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import joblib,xgboost as xgb
from scripts.frozen_pair_probes import partitions
from scripts.frozen_trajectory_probes import metrics
from scripts.evaluate_checkpoint import check_fold

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--cache',default='results/pair_probes');p.add_argument('--fold',type=int,choices=range(1,6),required=True);p.add_argument('--split',choices=['train','val','test'],default='test');p.add_argument('--output',required=True);a=p.parse_args()
    path=Path(a.checkpoint);check_fold({},path,a.fold)
    record=json.loads(path.with_suffix('.json').read_text());obj=joblib.load(path)
    data=np.load(Path(a.cache)/f'fold_{a.fold}_features.npz');ids=partitions(len(data['labels']),a.fold)[a.split];y=data['labels'][ids]
    pieces=[]
    if record['features'] in ['embedding','combined']:pieces.append(data['embeddings'][ids])
    if record['features'] in ['spectral','combined']:
        with np.load(Path(a.cache)/'spectral_features.npz') as f:
            assert np.array_equal(f['pids'],data['pids']);pieces.append(f['spectral'][ids])
    x=np.concatenate(pieces,axis=1)
    if record['family']=='logistic':
        assert np.array_equal(obj['model'].classes_,np.arange(20));prob=obj['model'].predict_proba(obj['scaler'].transform(x))
    else:prob=obj['model'].predict(xgb.DMatrix(x),iteration_range=(0,record['detail']['best_round']))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',indices=ids,pids=data['pids'][ids],labels=y,probabilities=prob,fold=a.fold,split=a.split,partial=False)
    r=dict(fold=a.fold,split=a.split,checkpoint=str(path),metrics=metrics(y,prob.argmax(1)))
    (out/'metrics.json').write_text(json.dumps(r,indent=2)+'\n');print({k:100*r['metrics'][k] for k in ['miou','oa','mean_recall']})
if __name__=='__main__':main()

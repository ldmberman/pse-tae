"""Average saved probabilities for one fold/split, verifying IDs and labels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.frozen_pair_probes import partitions
from scripts.frozen_trajectory_probes import metrics

def combine(paths,fold,split,cache,weights=None):
    with np.load(Path(cache)/f'fold_{fold}_features.npz') as f:
        ids=partitions(len(f['labels']),fold)[split];labels=f['labels'][ids];pids=f['pids'][ids]
    weights=np.ones(len(paths),float) if weights is None else np.asarray(weights,float)
    if len(weights)!=len(paths) or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:raise ValueError('Invalid mixture weights')
    weights/=weights.sum();total=np.zeros((len(ids),20),float)
    for path,w in zip(paths,weights):
        with np.load(path) as f:
            if 'fold' in f and f['fold'].item()!=fold:raise ValueError('Cross-fold ensemble member')
            if 'split' in f and f['split'].item()!=split:raise ValueError('Wrong member split')
            if 'partial' in f and f['partial'].item():raise ValueError('Partial predictions are not benchmark members')
            if not np.array_equal(f['indices'],ids) or not np.array_equal(f['labels'],labels):raise ValueError('Member IDs/labels do not match the requested fold/split')
            if 'probabilities' in f:p=f['probabilities'].astype(float)
            else:
                z=f['logits'].astype(float);z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
        if p.shape!=total.shape or not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid probability array')
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6);total+=w*p
    return ids,labels,pids,total,weights

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--members',nargs='+',required=True);p.add_argument('--weights',nargs='+',type=float);p.add_argument('--fold',type=int,choices=range(1,6),required=True);p.add_argument('--split',choices=['train','val','test'],default='test');p.add_argument('--cache',default='results/pair_probes');p.add_argument('--output',required=True);a=p.parse_args()
    ids,y,pids,prob,w=combine(a.members,a.fold,a.split,a.cache,a.weights);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',indices=ids,labels=y,pids=pids,probabilities=prob,fold=a.fold,split=a.split,partial=False)
    r=dict(fold=a.fold,split=a.split,members=a.members,weights=w.tolist(),metrics=metrics(y,prob.argmax(1)))
    (out/'metrics.json').write_text(json.dumps(r,indent=2)+'\n');print({k:100*r['metrics'][k] for k in ['miou','oa','mean_recall']})
if __name__=='__main__':main()

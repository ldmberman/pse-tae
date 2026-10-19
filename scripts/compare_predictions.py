"""Paired parcel bootstrap for mIoU(B)-mIoU(A), conditional on fitted models."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.frozen_trajectory_probes import metrics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('a');p.add_argument('b');p.add_argument('--output',required=True);p.add_argument('--replicates',type=int,default=1000);p.add_argument('--seed',type=int,default=20261018);a=p.parse_args()
    arrays=[];reference=None;metadata=None
    for path in [a.a,a.b]:
        with np.load(path) as f:
            if 'partial' in f and f['partial'].item():raise ValueError('Partial samples are not full test comparisons')
            meta={k:f[k].item() for k in ['fold','split'] if k in f}
            if metadata is not None and meta!=metadata:raise ValueError('Fold/split mismatch')
            metadata=meta
            ids=f['indices'];y=f['labels']
            if reference is not None:
                assert np.array_equal(reference[0],ids) and np.array_equal(reference[1],y)
            reference=(ids,y);arrays.append(f['probabilities' if 'probabilities' in f else 'logits'].argmax(1))
    rng=np.random.RandomState(a.seed);delta=[]
    for _ in range(a.replicates):
        ix=rng.randint(0,len(y),len(y));delta.append(metrics(y[ix],arrays[1][ix])['miou']-metrics(y[ix],arrays[0][ix])['miou'])
    result=dict(delta_points=100*(metrics(y,arrays[1])['miou']-metrics(y,arrays[0])['miou']),ci95_points=(100*np.quantile(delta,[.025,.975])).tolist(),replicates=a.replicates,seed=a.seed,caveat='Conditional on fitted models; not fold, seed, spatial or selection uncertainty.')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

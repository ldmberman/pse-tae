"""Rebuild fixed historical ensembles; never train or search on test.

Default: reload each checkpoint. --from-cache verifies and reuses the exact
archived probabilities to reproduce the recorded scores without model inference.
"""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.evaluate_checkpoint import digest
from scripts.ensemble_predictions import combine
from scripts.frozen_trajectory_probes import metrics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('ensemble');p.add_argument('--manifest',default='configs/ensembles.json');p.add_argument('--fold',type=int,required=True);p.add_argument('--split',choices=['train','val','test'],default='test');p.add_argument('--from-cache',action='store_true');p.add_argument('--cache',default='results/pair_probes');p.add_argument('--native',default='data/S2-2017-T31TFM-PixelSet');p.add_argument('--patch',default='data/s2-2017-IGARSS-NNI-NPY');p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=4);a=p.parse_args()
    config=json.loads(Path(a.manifest).read_text());assert config['fold']==a.fold,'This recipe has checkpoints only for its own fold'
    names=config['ensembles'][a.ensemble];paths=[]
    for name in names:
        m=config['members'][name];ck=Path(m['checkpoint']);assert digest(ck)==m['sha256'],f'Checkpoint changed: {name}'
        out=Path('outputs/ensemble_members')/f'Fold_{a.fold}'/name/a.split;out.mkdir(parents=True,exist_ok=True);pred=out/'predictions.npz'
        if a.from_cache:
            if a.split not in m['cached']:raise ValueError('No saved predictions for this split; omit --from-cache')
            cached=m['cached'][a.split];assert digest(cached['path'])==cached['sha256'],f'Cached input changed: {name}'
            with np.load(cached['path']) as f:
                arrays=dict(indices=f['indices'],labels=f['labels'],fold=a.fold,split=a.split,partial=False)
                arrays['logits' if cached['key']=='logits' else 'probabilities']=f[cached['key']]
                if 'probabilities' in arrays:
                    q=arrays['probabilities'].astype(float);arrays['probabilities']=q/q.sum(1,keepdims=True)
                np.savez_compressed(pred,**arrays)
        else:
            cmd=[sys.executable,'scripts/evaluate_tabular.py' if m['family']=='tabular' else 'scripts/evaluate_checkpoint.py','--checkpoint',str(ck),'--fold',str(a.fold),'--split',a.split,'--cache',a.cache,'--output',str(out)]
            if m['family']!='tabular':cmd+=['--family',m['family'],'--variant',m['variant'],'--dataset',a.native,'--native',a.native,'--patch',a.patch,'--device',a.device,'--batch-size',str(a.batch_size)]
            subprocess.run(cmd,check=True)
        paths.append(str(pred))
    ids,y,pids,prob,weights=combine(paths,a.fold,a.split,a.cache)
    out=Path('outputs/ensembles')/a.ensemble/a.split;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',indices=ids,labels=y,pids=pids,probabilities=prob,fold=a.fold,split=a.split,partial=False)
    r=dict(fold=a.fold,split=a.split,manifest_sha256=digest(a.manifest),from_cache=a.from_cache,members=names,weights=weights.tolist(),metrics=metrics(y,prob.argmax(1)))
    (out/'metrics.json').write_text(json.dumps(r,indent=2)+'\n');print({k:100*r['metrics'][k] for k in ['miou','oa','mean_recall']})
if __name__=='__main__':main()

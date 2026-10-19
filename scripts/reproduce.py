"""Portable entry points for the research history. Use --dry-run to inspect commands."""
import argparse,json,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def command(a):
    recipes=json.loads((ROOT/'configs/experiments.json').read_text());r=recipes[a.experiment]
    fold=a.fold;out=r['output'];kind=r['kind'];variant=r['variant'];baseline=a.baseline or f'results/baseline/Fold_{fold}/model.pth.tar'
    replace={'baseline':baseline,'capacity':a.source or f'results/grouped_scratch/Fold_{fold}/capacity/best.pt'}
    flags=[x.format(**replace) for x in r.get('train_args',[])]
    ck=a.checkpoint or f'{out}/Fold_{fold}/'+('model.pth.tar' if kind=='native' else f'{variant}/best.pt')
    if a.action=='plots':return [sys.executable,'scripts/plot_predictions.py',f'outputs/{a.experiment}/Fold_{fold}/{a.split}/predictions.npz','--output',f'outputs/{a.experiment}/Fold_{fold}/{a.split}/plots']
    if a.action=='evaluate':
        cmd=[sys.executable,'scripts/evaluate_checkpoint.py','--checkpoint',ck,'--family',r['family'],'--variant',variant,'--fold',str(fold),'--split',a.split,'--dataset',a.native,'--native',a.native,'--patch',a.patch,'--cache',a.cache,'--device',a.device,'--workers',str(a.workers),'--batch-size',str(a.batch_size or 4),'--output',f'outputs/{a.experiment}/Fold_{fold}/{a.split}']
        if kind=='native':cmd+=['--experiment',a.experiment]
        if a.experiment=='moments':cmd+=['--legacy-unscaled-kurtosis']
        if a.limit:cmd+=['--limit',str(a.limit)]
        return cmd
    if kind=='native':
        cmd=[sys.executable,'train.py','--dataset_folder',a.native,'--res_dir',out,'--start_fold',str(fold),'--fold_count',str(fold),'--num_workers',str(a.workers),'--device',a.device]+flags
        if a.epochs:cmd+=['--epochs',str(a.epochs)]
        if a.batch_size:cmd+=['--batch_size',str(a.batch_size)]
        return cmd
    cmd=[sys.executable,'scripts/'+r['script']]
    if kind!='distill':cmd+=['fit' if kind=='small' else 'train']
    cmd+=['--variant',variant,'--fold',str(fold),('--out' if kind=='pixel' else '--output'),out,'--device',a.device]
    if r['family']!='small_correction':cmd+=['--workers',str(a.workers)]
    if kind=='pixel':cmd+=['--features',a.cache,'--pixels','results/frozen_trajectories/native_cache','--native',a.native]
    else:
        cmd+=['--cache',a.cache]
        if r['family']!='small_correction':cmd+=['--native',a.native]
    if kind not in ['pixel','distill','small']:cmd+=['--patch',a.patch]
    if kind=='small':cmd+=['--trajectories','results/frozen_trajectories/native_cache','--patch-trajectories','results/frozen_trajectories/patch_cache']
    if kind=='distill':
        if not a.teacher:raise ValueError('--teacher must name same-fold TRAIN probabilities')
        cmd+=['--teacher',a.teacher]
    if a.epochs:cmd+=['--epochs',str(a.epochs)]
    if a.batch_size:cmd+=['--batch-size',str(a.batch_size)]
    return cmd+flags

def main():
    recipes=json.loads((ROOT/'configs/experiments.json').read_text())
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',choices=recipes);p.add_argument('action',choices=['train','evaluate','plots']);p.add_argument('--fold',type=int,choices=range(1,6),default=1);p.add_argument('--split',choices=['train','val','test'],default='test');p.add_argument('--native',default='data/S2-2017-T31TFM-PixelSet');p.add_argument('--patch',default='data/s2-2017-IGARSS-NNI-NPY');p.add_argument('--cache',default='results/pair_probes');p.add_argument('--device',default=None);p.add_argument('--workers',type=int,default=0);p.add_argument('--batch-size',type=int);p.add_argument('--epochs',type=int);p.add_argument('--checkpoint');p.add_argument('--baseline');p.add_argument('--source');p.add_argument('--teacher');p.add_argument('--limit',type=int,default=0);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    a.device=a.device or ('cuda' if a.action=='train' else 'cpu');cmd=command(a);print(shlex.join(cmd),flush=True)
    if not a.dry_run:subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__':main()

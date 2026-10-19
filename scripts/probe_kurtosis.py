"""New deterministic own-fold kurtosis measurements with ROC-AUC and PR metrics.

All available pixels are used, in bounded chunks. Constant date/channel embeddings
have undefined kurtosis and are excluded from the parcel average. The sample is
balanced by class; its AP/PR-AUC are not estimates at population class prevalence.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
from scipy.stats import spearmanr


def embedding_excess(encoder, raw, mean, std, chunk=2048):
    import torch
    n=raw.shape[-1]
    def hidden(start):
        x=((raw[...,start:start+chunk].astype(np.float32)-mean[...,None])/std[...,None]).astype(np.float32)
        return encoder(torch.from_numpy(x)).numpy().astype(np.float64)
    with torch.inference_mode():
        total=np.zeros((raw.shape[0],64),np.float64)
        for start in range(0,n,chunk):total+=hidden(start).sum(-1)
        center=total/n;second=np.zeros_like(center);fourth=np.zeros_like(center)
        for start in range(0,n,chunk):
            d=hidden(start)-center[...,None];d2=d*d
            second+=d2.sum(-1);fourth+=(d2*d2).sum(-1)
    variance=second/n;valid=variance>1e-12
    if not valid.any():raise ValueError('No nonconstant embedding channels')
    excess=(fourth/n)[valid]/variance[valid]**2-3
    return float(excess.mean()),int(valid.sum()),int(valid.size)


def summarize(rows):
    y=np.array([r['label']==0 for r in rows]);score=np.array([r['kurtosis'] for r in rows])
    precision,recall,_=precision_recall_curve(y,score)
    pixels=np.array([r['pixels'] for r in rows])
    size_control=dict(roc_auc=float(roc_auc_score(y,pixels)),average_precision=float(average_precision_score(y,pixels)),
                      spearman_with_kurtosis=float(spearmanr(pixels,score).statistic),
                      median_pixels_meadow=float(np.median(pixels[y])),median_pixels_non_agricultural=float(np.median(pixels[~y])))
    return dict(n=len(rows),meadow=int(y.sum()),non_agricultural=int((~y).sum()),
                roc_auc=float(roc_auc_score(y,score)),average_precision=float(average_precision_score(y,score)),
                pr_auc_trapezoid=float(auc(recall,precision)),
                median_meadow=float(np.median(score[y])),median_non_agricultural=float(np.median(score[~y])),pixel_count_control=size_control)


def plot(rows,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y=np.array([r['label']==0 for r in rows]);score=np.array([r['kurtosis'] for r in rows])
    fpr,tpr,_=roc_curve(y,score);precision,recall,_=precision_recall_curve(y,score)
    for name,x,v,xlabel,ylabel,label in [
        ('roc',fpr,tpr,'False-positive rate','True-positive rate','Excess kurtosis'),
        ('pr',recall,precision,'Recall','Precision','Excess kurtosis')]:
        fig,ax=plt.subplots(figsize=(6.2,4.4),layout='constrained')
        ax.plot(x,v,color='#176a9c',label=label)
        pixels=np.array([r['pixels'] for r in rows])
        if name=='roc':px,py,_=roc_curve(y,pixels);plabel='Native pixel count'
        else:py,px,_=precision_recall_curve(y,pixels);plabel='Native pixel count'
        ax.plot(px,py,color='#b55b26',ls='--',label=plabel)
        if name=='roc':ax.plot([0,1],[0,1],'--',color='#999')
        else:ax.axhline(y.mean(),ls='--',color='#999',label='Sample prevalence')
        ax.set(xlabel=xlabel,ylabel=ylabel,xlim=(0,1),ylim=(0,1.02),title='Embedding excess kurtosis: Meadow vs Non-agricultural')
        ax.legend(fontsize=9);ax.grid(alpha=.15);fig.savefig(out/f'{name}.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.2,4.4),layout='constrained')
    for cls,label in [(True,'Meadow'),(False,'Non-agricultural')]:ax.hist(score[y==cls],bins=35,density=True,alpha=.5,label=label)
    ax.set(xlabel='Mean embedding excess kurtosis',ylabel='Density');ax.legend();fig.savefig(out/'distribution.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='data/S2-2017-T31TFM-PixelSet');p.add_argument('--baseline',default='results/baseline')
    p.add_argument('--per-class',type=int,default=120);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--plots-only',action='store_true',help='Recompute metrics and plots from saved scores without encoder inference')
    p.add_argument('--output',type=Path,default=Path('outputs/kurtosis'));a=p.parse_args()
    if a.per_class<1:p.error('--per-class must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.plots_only:
        result=json.loads((a.output/'metrics.json').read_text());rows=result['records']
        result['pooled']=summarize(rows);result['per_fold']=[dict(fold=f,**summarize([r for r in rows if r['fold']==f])) for f in range(1,6)]
        (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');plot(rows,a.output);return
    import torch
    from scripts.frozen_pair_probes import make_dataset, partitions
    from models.stclassifier import PseTae
    torch.set_num_threads(a.threads)
    (a.output/'samples.jsonl').write_text('')
    ds=make_dataset(a);y=np.asarray(ds.target);mean,std=ds.norm;rows=[];provenance=[];start=time.time()
    for fold in range(1,6):
        checkpoint=Path(a.baseline)/f'Fold_{fold}'/'model.pth.tar'
        model=PseTae(positions=ds.date_positions);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']);model.eval()
        provenance.append(dict(fold=fold,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        test=partitions(len(y),fold)['test'];rng=np.random.RandomState(1+fold)
        for cls in [0,18]:
            candidates=test[y[test]==cls].copy();rng.shuffle(candidates);count=0
            for i in candidates:
                raw=np.load(Path(a.dataset)/'DATA'/f'{ds.pid[i]}.npy',mmap_mode='r')
                if raw.shape[-1]<6:continue
                value,valid,total=embedding_excess(model.spatial_encoder.mlp1,raw,mean,std)
                row=dict(fold=fold,index=int(i),pid=str(ds.pid[i]),label=int(cls),pixels=raw.shape[-1],kurtosis=value,valid_date_channels=valid,date_channels=total)
                rows.append(row);count+=1
                with (a.output/'samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if count>=a.per_class:break
            print('FOLD',fold,'class',cls,'sample',count,'seconds',round(time.time()-start),flush=True)
    result=dict(protocol=dict(sample='up to 120 per class per fold by default; >=6 pixels; own-fold test only',per_class=a.per_class,
                              seed='1 + fold',positive='Meadow',negative='Non-agricultural',variance='population m2',
                              constant_channels='exclude m2 <= 1e-12',aggregation='mean excess kurtosis across valid dates/channels',
                              checkpoint_provenance=provenance),pooled=summarize(rows),
                per_fold=[dict(fold=f,**summarize([r for r in rows if r['fold']==f])) for f in range(1,6)],records=rows,seconds=time.time()-start)
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');plot(rows,a.output)
    print(json.dumps(result['pooled']),flush=True)
if __name__=='__main__':main()

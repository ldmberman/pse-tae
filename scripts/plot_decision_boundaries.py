"""Validation-selected decision rules, compared on one held-out test fold.

ROC uses weak-class margin against the strongest alternative, so the marked zero
threshold reproduces the actual multiclass decision. Recorded mIoU uses all 20
classes, not just the binary ROC. No threshold is selected on test labels.
"""
import argparse,json,sys,shutil,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import joblib
from scipy.special import expit
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score
from learning.splits import partitions


def metrics(y,pred):
    cm=np.bincount(y*20+pred,minlength=400).reshape(20,20);tp=cm.diagonal()
    return dict(oa=float(tp.sum()/cm.sum()),miou=float(np.mean(tp/np.maximum(cm.sum(0)+cm.sum(1)-tp,1))),mean_recall=float(np.mean(tp/np.maximum(cm.sum(1),1))))


def margin(z,weak):
    others=z.copy();others[:,weak]=-np.inf
    return z[:,weak]-others.max(1)


def load_scores(path,ids,y):
    d=np.load(path);assert np.array_equal(d['indices'],ids),path
    if 'labels' in d:assert np.array_equal(d['labels'],y),path
    return d['logits']


def plot_report(report,out):
    rows=report['methods']
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,name in zip(axes.flat,['tau','weighted','frozen_head','spectral_logistic']):
        for key,color in [('baseline','#777'),(name,'#176a9c')]:
            r=rows[key];ax.plot(r['curve']['fpr'],r['curve']['tpr'],color=color,label='Baseline' if key=='baseline' else 'Method')
        r=rows[name];pt=r['point'];ax.scatter([pt['fpr']],[pt['tpr']],s=55,color='#b04724',zorder=4)
        ax.annotate("Selected rule",(pt['fpr'],pt['tpr']),xytext=(.40,.12),textcoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#b04724'),fontsize=9)
        ax.plot([0,1],[0,1],'--',color='#bbb',lw=.8);ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False-positive rate',ylabel='Recall',title=r['label']);ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.12)
    fig.suptitle(f'Fold {report['fold']} test · Winter Durum Wheat vs all other classes\n{report['weak_count']} positives among {report['n']} parcels · annotated mIoU uses all 20 classes',fontsize=12)
    fig.savefig(out/'roc.svg');fig.savefig(out/'roc.png',dpi=140);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=Path('results/pair_probes'));p.add_argument('--fold',type=int,default=1)
    p.add_argument('--weak',type=int,default=1);p.add_argument('--confounder',type=int,default=4)
    p.add_argument('--weighted',type=Path,default=Path('outputs/readme_diagnostics/class_weighted/predictions.npz'))
    p.add_argument('--frozen',type=Path,default=Path('results/frozen_trajectories'));p.add_argument('--frozen-scores',type=Path,help='Replacement-head predictions from evaluate_checkpoint.py')
    p.add_argument('--plots-only',action='store_true',help='Render saved metrics without checkpoint inference or feature loading')
    p.add_argument('--output',type=Path,default=Path('outputs/decision_boundaries'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.plots_only:
        plot_report(json.loads((a.output/'metrics.json').read_text()),a.output);return
    d=np.load(a.cache/f'fold_{a.fold}_features.npz');y=d['labels'];z=d['logits'];split=partitions(len(y),a.fold);va,te=split['val'],split['test'];truth=y[te]
    pi=np.bincount(y[split['train']],minlength=20)/len(split['train']);logpi=np.log(pi)
    sweep=[]
    for tau in np.linspace(0,2.5,51):sweep.append(dict(tau=float(tau),**metrics(y[va],(z[va]-tau*logpi).argmax(1))))
    selected=max(sweep,key=lambda r:r['miou']);tau=selected['tau'];adjusted=z[te]-tau*logpi
    base_pred=z[te].argmax(1)
    base_margin=margin(z[te],a.weak)
    methods={'baseline':dict(label='Baseline',pred=base_pred,score=base_margin),
             'tau':dict(label=f'Logit adjustment · τ={tau:g}',pred=adjusted.argmax(1),score=margin(adjusted,a.weak))}
    w=load_scores(a.weighted,te,truth);methods['weighted']=dict(label='Class-weighted focal loss · γ=1',pred=w.argmax(1),score=margin(w,a.weak))
    h=load_scores(a.frozen_scores or a.frozen/f'Fold_{a.fold}/control/test_predictions.npz',te,truth)
    methods['frozen_head']=dict(label='New frozen-embedding head · γ=1',pred=h.argmax(1),score=margin(h,a.weak))
    # Memory-map the large spectral matrix rather than materializing the NPZ array.
    matrix_file=a.output/'spectral.npy'
    if not matrix_file.exists():
        with zipfile.ZipFile(a.cache/'spectral_features.npz') as archive,archive.open('spectral.npy') as source,matrix_file.open('wb') as target:shutil.copyfileobj(source,target)
    x=np.load(matrix_file,mmap_mode='r')
    fitted=joblib.load(a.cache/f'Fold_{a.fold}/{a.weak}_{a.confounder}/spectral_logistic.joblib')
    def spectral_scores(ids):
        result=[]
        for start in range(0,len(ids),1024):
            xx=fitted['scaler'].transform(x[ids[start:start+1024]])
            result.append(fitted['model'].decision_function(xx))
        return np.concatenate(result)
    sv=spectral_scores(va);st=spectral_scores(te);bv=z[va].argmax(1);vgate=np.isin(bv,[a.weak,a.confounder])
    best=dict(enabled=False,threshold=None,**metrics(y[va],bv))
    for t in np.unique(np.quantile(sv[vgate],np.linspace(0,1,201))):
        pred=bv.copy();pred[vgate]=np.where(sv[vgate]>=t,a.weak,a.confounder);met=metrics(y[va],pred)
        if met['miou']>best['miou']:best=dict(enabled=True,threshold=float(t),**met)
    pred=base_pred.copy();gate=np.isin(pred,[a.weak,a.confounder])
    if best['enabled']:
        pred[gate]=np.where(st[gate]>=best['threshold'],a.weak,a.confounder)
        score=np.where(gate,expit(st-best['threshold'])-.5,-1.)
    else:score=base_margin.copy()
    methods['spectral_logistic']=dict(label='Spectral logistic pair rerouting',pred=pred,score=score)
    binary=truth==a.weak;rows={};arrays=dict(indices=te,labels=truth)
    for name,item in methods.items():
        assert np.array_equal(item['score']>=0,item['pred']==a.weak),'Operating point must equal the deployed decision'
        pred=item['pred']==a.weak;tp=int(np.sum(binary&pred));fp=int(np.sum(~binary&pred))
        fpr,tpr,_=roc_curve(binary,item['score'])
        rows[name]=dict(label=item['label'],**metrics(truth,item['pred']),roc_auc=float(roc_auc_score(binary,item['score'])),
                        average_precision=float(average_precision_score(binary,item['score'])),
                        point=dict(fpr=fp/int((~binary).sum()),tpr=tp/int(binary.sum()),tp=tp,fp=fp),
                        curve=dict(fpr=fpr.tolist(),tpr=tpr.tolist()))
        arrays[name+'_margin']=item['score'];arrays[name+'_pred']=item['pred']
    report=dict(fold=a.fold,weak=a.weak,confounder=a.confounder,n=len(te),weak_count=int(binary.sum()),methods=rows,tau_sweep=sweep,
                selected_tau=selected,spectral_validation_choice=best,
                note='New fixed-pixel diagnostic evaluation. ROC is weak class vs all others; marked decisions use validation rules; mIoU spans 20 classes.')
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output/'scores.npz',**arrays)
    plot_report(report,a.output)
    print(json.dumps({k:{m:v[m] for m in ['miou','oa','mean_recall','roc_auc']} for k,v in rows.items()}),flush=True)
if __name__=='__main__':main()

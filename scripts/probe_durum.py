"""New fold-specific moment probes and NDVI curves for Durum vs Winter Cereal.

Each model sees per-date/per-band moments, standardized on pair training parcels.
C is selected on validation AP; test is evaluated once after that choice.
"""
import argparse,json,sys,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve,auc
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import joblib
from learning.splits import partitions


def metrics(y,s):
    p,r,_=precision_recall_curve(y,s)
    return dict(roc_auc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)),pr_auc_trapezoid=float(auc(r,p)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,default=Path('results/xgboost'))
    p.add_argument('--plots-only',action='store_true',help='Render the saved metrics and scores without refitting');p.add_argument('--evaluate',action='store_true',help='Load each saved model and recompute test scores without fitting');p.add_argument('--calendar',type=Path,required=True);p.add_argument('--fold',type=int,default=1);p.add_argument('--output',type=Path,default=Path('outputs/durum_moments'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    y=np.load(a.features/'targets.npy');mom=np.load(a.features/'moments.npy',mmap_mode='r');split=partitions(len(y),a.fold)
    ix={k:v[np.isin(y[v],[1,4])] for k,v in split.items()};ys={k:(y[v]==1).astype(int) for k,v in ix.items()}
    counts={k:dict(durum=int(v.sum()),winter_cereal=int((v==0).sum())) for k,v in ys.items()}
    scores={};records=[]
    for k in ([] if a.plots_only else range(1,6)):
        name=f'moments_1_to_{k}';dest=a.output/name;dest.mkdir(exist_ok=True)
        features={s:np.asarray(mom[v,:,:,:k]).reshape(len(v),-1) for s,v in ix.items()}
        if a.evaluate:
            fitted=joblib.load(dest/'model.joblib');s=fitted['model'].decision_function(fitted['scaler'].transform(features['test']));scores[name]=s
            previous=json.loads((a.output/'metrics.json').read_text())['models'][k-1];records.append(dict(previous,test=metrics(ys['test'],s)));continue
        scaler=StandardScaler().fit(features['train']);features={s:scaler.transform(x) for s,x in features.items()}
        best=None;trials=[]
        for c in [.01,.1,1.]:
            model=LogisticRegression(C=c,class_weight='balanced',solver='lbfgs',max_iter=1000,tol=1e-4)
            with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning);model.fit(features['train'],ys['train'])
            v=model.decision_function(features['val']);ap=average_precision_score(ys['val'],v)
            converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
            trials.append(dict(C=c,validation_ap=float(ap),converged=converged))
            if best is None or ap>best[0]:best=(ap,c,model,converged)
        _,c,model,converged=best;s=model.decision_function(features['test']);scores[name]=s
        joblib.dump(dict(model=model,scaler=scaler),dest/'model.joblib')
        records.append(dict(name=name,features=240*k,C=c,converged=converged,validation_trials=trials,test=metrics(ys['test'],s)))
        print(name,json.dumps(records[-1]),flush=True)
    if a.plots_only:
        saved=np.load(a.output/'scores.npz');assert np.array_equal(saved['indices'],ix['test']);assert np.array_equal(saved['labels'],ys['test'])
        records=json.loads((a.output/'metrics.json').read_text())['models'];scores={r['name']:saved[r['name']] for r in records}
    else:np.savez_compressed(a.output/'scores.npz',labels=ys['test'],indices=ix['test'],**scores)
    result=dict(fold=a.fold,positive='Winter Durum Wheat',negative='Winter Cereal',counts=counts,
                protocol='Own-fold 60/20/20; balanced logistic regression; C selected on validation AP; no test selection',models=records)
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for mode in ['roc','pr']:
        fig,ax=plt.subplots(figsize=(6.7,4.8),layout='constrained')
        for row in records:
            s=scores[row['name']];label=f"First {row['features']//240} moments"
            if mode=='roc':x,v,_=roc_curve(ys['test'],s)
            else:v,x,_=precision_recall_curve(ys['test'],s)
            ax.plot(x,v,label=label,lw=1.5)
        if mode=='roc':ax.plot([0,1],[0,1],'--',color='#999')
        else:ax.axhline(ys['test'].mean(),ls='--',color='#999',label="Sample prevalence")
        ax.set(xlabel='False-positive rate' if mode=='roc' else 'Recall',ylabel='True-positive rate' if mode=='roc' else 'Precision',xlim=(0,1),ylim=(0,1.02),title=f'Durum vs Winter Cereal · fold {a.fold} test')
        ax.legend(fontsize=9);ax.grid(alpha=.15);fig.savefig(a.output/f'{mode}.svg');plt.close(fig)
    calendar=json.loads(a.calendar.read_text());calendar={'dates':calendar.get('dates',[calendar[str(i)] for i in range(len(calendar))] if 'dates' not in calendar else [])};dates=np.array([np.datetime64(f'{d[:4]}-{d[4:6]}-{d[6:]}') for d in map(str,calendar['dates'])])
    mu=np.asarray(mom[ix['test'],:,:,0]);num=mu[:,:,6]-mu[:,:,2];den=mu[:,:,6]+mu[:,:,2]
    ndvi=np.divide(num,den,out=np.full_like(num,np.nan),where=den>1e-6)
    fig,ax=plt.subplots(figsize=(7.5,4.6),layout='constrained');curves={}
    for c,label,color in [(1,'Winter Durum Wheat','#a64623'),(0,'Winter Cereal','#176a9c')]:
        values=ndvi[ys['test']==c];low,med,high=np.nanpercentile(values,[25,50,75],axis=0)
        ax.plot(dates,med,color=color,label=f'{label} · {len(values)} parcels');ax.fill_between(dates,low,high,color=color,alpha=.16)
        curves[label]=dict(n=len(values),valid_per_date=np.isfinite(values).sum(0).tolist(),median=med.tolist(),q25=low.tolist(),q75=high.tolist())
    ax.set(xlabel='Acquisition date',ylabel='NDVI = (B8 − B4) / (B8 + B4)',title=f'Fold {a.fold} test · parcel-mean reflectance · median and middle 50%');ax.legend();ax.grid(alpha=.15)
    fig.savefig(a.output/'ndvi.svg');fig.savefig(a.output/'ndvi.png',dpi=140);plt.close(fig)
    (a.output/'ndvi.json').write_text(json.dumps(dict(fold=a.fold,dates=calendar['dates'],curves=curves),indent=2)+'\n')
if __name__=='__main__':main()

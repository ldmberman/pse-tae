"""Recompute metrics, confusion matrix, ROC-AUC/AP and curves from saved predictions."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.frozen_pair_probes import NAMES
from scripts.frozen_trajectory_probes import metrics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('predictions');p.add_argument('--output',required=True);a=p.parse_args()
    with np.load(a.predictions) as f:
        y=f['labels'];meta={k:f[k].item() for k in ['fold','split','partial'] if k in f}
        if 'probabilities' in f:probs=f['probabilities'].astype(float)
        else:
            z=f['logits'].astype(float);z-=z.max(1,keepdims=True);probs=np.exp(z);probs/=probs.sum(1,keepdims=True)
    assert probs.shape==(len(y),20) and np.isfinite(probs).all()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);pred=probs.argmax(1)
    cm=np.bincount(20*y+pred,minlength=400).reshape(20,20);m=metrics(y,pred)
    (out/'metrics.json').write_text(json.dumps(dict(**meta,n=len(y),classes=NAMES,metrics=m,confusion=cm.tolist()),indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(12,10),layout='constrained');norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
    im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues');ax.set(xticks=range(20),yticks=range(20),xticklabels=NAMES,yticklabels=NAMES,xlabel='Predicted class',ylabel='True class',title='Class recall (row-normalized confusion matrix)'+(' — PARTIAL SAMPLE' if meta.get('partial') else ''))
    plt.setp(ax.get_xticklabels(),rotation=65,ha='right');fig.colorbar(im,ax=ax);fig.savefig(out/'confusion.svg');plt.close(fig)
    fig,axes=plt.subplots(5,4,figsize=(16,16),layout='constrained');rows=[]
    rocfig,rocaxes=plt.subplots(5,4,figsize=(16,16),layout='constrained')
    for c,(ax,rax) in enumerate(zip(axes.flat,rocaxes.flat)):
        truth=y==c
        if not truth.any() or truth.all():
            ax.set_title(NAMES[c]+' (AUC undefined)');rax.set_title(NAMES[c]+' (AUC undefined)');continue
        pr,re,_=precision_recall_curve(truth,probs[:,c]);fpr,tpr,_=roc_curve(truth,probs[:,c]);ap=average_precision_score(truth,probs[:,c]);auc=roc_auc_score(truth,probs[:,c])
        ax.step(re,pr,where='post');ax.axhline(truth.mean(),ls='--',color='grey');ax.set(title=f'{NAMES[c]}\nAP {ap:.3f}',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
        rax.plot(fpr,tpr);rax.plot([0,1],[0,1],'--',color='grey');rax.set(title=f'{NAMES[c]}\nROC-AUC {auc:.3f}',xlabel='False positive rate',ylabel='Recall')
        rows.append(dict(class_name=NAMES[c],positive=int(truth.sum()),roc_auc=auc,average_precision=ap,iou=m['iou'][c],recall=m['recall'][c]))
    fig.savefig(out/'precision_recall.svg');rocfig.savefig(out/'roc.svg');plt.close('all')
    with (out/'per_class.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['class_name','positive','roc_auc','average_precision','iou','recall']);w.writeheader();w.writerows(rows)
    pairs=sorted([(int(cm[i,j]),NAMES[i],NAMES[j]) for i in range(20) for j in range(20) if i!=j],reverse=True)
    (out/'top_confusions.json').write_text(json.dumps([dict(count=n,true=t,predicted=p) for n,t,p in pairs[:20]],indent=2)+'\n')
    print(out)
if __name__=='__main__':main()

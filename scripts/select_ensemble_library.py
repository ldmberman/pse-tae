"""Validation-only bounded ensemble selection, followed by frozen test evaluation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ensemble_library/Fold_1';REQ=ROOT/'results/requested_comparison';MC=ROOT/'results/multiclass_probes/Fold_1';SNAP=ROOT/'results/complementarity/snapshot_20261018_1605';CONTROL=ROOT/'results/checkpoint_control/Fold_1'
KEYS=['miou','oa','mean_recall']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def softmax(z):
 z=z.astype(float);z-=z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
def score(y,p):
 pred=p.argmax(1) if p.ndim==2 else p;cm=np.bincount(20*y+pred,minlength=400).reshape(20,20);t=np.diag(cm)
 return dict(miou=float(np.mean(t/np.maximum(cm.sum(0)+cm.sum(1)-t,1))),oa=float(t.sum()/cm.sum()),mean_recall=float(np.mean(t/np.maximum(cm.sum(1),1))),iou=(t/np.maximum(cm.sum(0)+cm.sum(1)-t,1)).tolist(),recall=(t/np.maximum(cm.sum(1),1)).tolist(),confusion=cm.tolist())
def summary(m):return {k:m[k] for k in KEYS}
def paths(name,split):
 if name in ['baseline_27','baseline_78']:return CONTROL/f"epoch_{int(name.split('_')[1]):03d}_{split}.npz"
 if name=='cnn_spatial':return REQ/f'Fold_1/cnn_spatial_{split}.npz'
 return OUT/f'{name}_{split}.npz'
def load(split,names):
 used={};tabpath=MC/f'{split}_scores.npz';used[str(tabpath.relative_to(ROOT))]=sha(tabpath)
 with np.load(tabpath) as f:
  ids=f['indices'];y=f['labels'];tab={n:f[n].astype(float) for n in names+['combined_xgb_sqrt_5'] if n in f.files}
 for n,p in tab.items():
  assert p.shape==(len(y),20) and np.isfinite(p).all() and (p>=0).all();np.testing.assert_allclose(p.sum(1),1,atol=1e-6,rtol=0);p/=p.sum(1,keepdims=True)
 def logits(path):
  used[str(path.relative_to(ROOT))]=sha(path)
  with np.load(path) as f:
   assert np.array_equal(ids,f['indices']) and np.array_equal(y,f['labels']),path
   z=f['logits'];assert z.shape==(len(y),20) and np.isfinite(z).all();return softmax(z)
 if split=='val':
  components=[logits(SNAP/n/'validation_predictions.npz') for n in ['baseline','capacity','grouped']]
  components += [logits(REQ/'Fold_1/date_dropout_val.npz'),logits(REQ/'cnn_temporal_snapshot_20261018/val_predictions.npz'),tab['combined_xgb_sqrt_5']]
  base=sum(components)/6
 else:
  path=REQ/'six_model_ensemble/test_predictions.npz';used[str(path.relative_to(ROOT))]=sha(path)
  with np.load(path) as f:assert np.array_equal(ids,f['indices']) and np.array_equal(y,f['labels']);base=f['probabilities'].astype(float)
 candidates={n:tab[n] if n in tab else logits(paths(n,split)) for n in names}
 return ids,y,base,candidates,used

def select():
 protocol=read(OUT/'protocol.json');assert protocol['fold']==1
 names=['baseline_27','baseline_78','native_baseline','cnn_spatial','capacity_large','ltae','rope']+[n for n in read(MC/'selection.json')['models'] if n!='combined_xgb_sqrt_5']
 ids,y,base,candidates,used=load('val',names);reference=score(y,base);rank=[]
 for n,p in candidates.items():
  m=score(y,(6*base+p)/7);rank.append(dict(name=n,standalone=summary(score(y,p)),addition=summary(m),delta=m['miou']-reference['miou']))
 rank.sort(key=lambda r:r['delta'],reverse=True);selected=[];total=6*base;current=reference['miou'];trace=[]
 for step in range(4):
  trials=[(score(y,(total+candidates[n])/(7+len(selected)))['miou'],n) for n in names if n not in selected]
  best,n=max(trials,key=lambda r:(r[0],r[1]));trace.append(dict(step=step+1,candidate=n,validation_miou=best,delta=best-current,accepted=best-current>=.0005))
  if best-current<.0005:break
  total+=candidates[n];selected.append(n);current=best
 equal=total/(6+len(selected));alpha=len(selected)/(6+len(selected));weight_trials=[dict(alpha=alpha,metrics=summary(score(y,equal)),kind='equal_members')];chosen=weight_trials[0]
 if selected:
  additions=sum(candidates[n] for n in selected)/len(selected)
  for a in [.1,.2,.3,.4,.5]:
   m=summary(score(y,(1-a)*base+a*additions));row=dict(alpha=a,metrics=m,kind='group_weight');weight_trials.append(row)
   if m['miou']>chosen['metrics']['miou']+1e-12:chosen=row
 result=dict(fold=1,created_unix=time.time(),protocol_sha256=sha(OUT/'protocol.json'),selector_sha256=sha(Path(__file__)),validation_source_sha256=used,reference=summary(reference),single_addition_ranking=rank,greedy_trace=trace,selected=selected,weight_trials=weight_trials,chosen=chosen,equal=summary(score(y,equal)),eight_control=summary(score(y,(6*base+candidates['baseline_27']+candidates['baseline_78'])/8)),n_validation=len(y))
 path=OUT/'selection.json'
 if path.exists():raise RuntimeError('Selection already frozen; use evaluate, do not rerun selection.')
 path.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['reference','single_addition_ranking','greedy_trace','selected','weight_trials','chosen','eight_control']},indent=2))

def evaluate():
 s=read(OUT/'selection.json');assert sha(OUT/'protocol.json')==s['protocol_sha256'];assert sha(Path(__file__))==s['selector_sha256']
 for f,h in s['validation_source_sha256'].items():assert sha(ROOT/f)==h
 names=list(dict.fromkeys(s['selected']+['baseline_27','baseline_78']));ids,y,base,c,used=load('test',names)
 with np.load(MC/'val_scores.npz') as f:assert not np.intersect1d(ids,f['indices']).size
 k=len(s['selected']);added=sum((c[n] for n in s['selected']),np.zeros_like(base));a=s['chosen']['alpha']
 prob={'six_model_reference':base,'eight_checkpoint_control':(6*base+c['baseline_27']+c['baseline_78'])/8,'greedy_equal':(6*base+added)/(6+k),'validation_selected':(1-a)*base+a*added/k if k else base}
 metrics={n:score(y,p) for n,p in prob.items()};prior=read(REQ/'six_model_ensemble/report.json')['metrics']['six_model_ensemble']
 for key in KEYS:assert abs(metrics['six_model_reference'][key]-prior[key])<1e-12
 for n,p in prob.items():
  np.testing.assert_allclose(p.sum(1),1,atol=1e-12,rtol=0);np.savez_compressed(OUT/f'{n}_test.npz',indices=ids,labels=y,probabilities=p,logits=np.log(p.clip(1e-300)))
 preds={n:p.argmax(1) for n,p in prob.items()};rng=np.random.RandomState(20261018);draws={n:[] for n in prob if n!='six_model_reference'}
 for _ in range(1000):
  ix=rng.randint(0,len(y),len(y));ref=score(y[ix],preds['six_model_reference'][ix])['miou']
  for n in draws:draws[n].append(score(y[ix],preds[n][ix])['miou']-ref)
 deltas={n:dict(delta=metrics[n]['miou']-prior['miou'],ci95=np.quantile(v,[.025,.975]).tolist()) for n,v in draws.items()}
 report=dict(fold=1,n_test=len(y),selection_sha256=sha(OUT/'selection.json'),test_source_sha256=used,selected=s['selected'],alpha=a,metrics=metrics,selected_member_test_metrics={n:score(y,c[n]) for n in s['selected']},paired_miou_differences=deltas,per_class_iou_delta=(np.array(metrics['validation_selected']['iou'])-np.array(prior['iou'])).tolist(),caveat=read(OUT/'protocol.json')['caveat'])
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({n:summary(m) for n,m in metrics.items()},indent=2));print(json.dumps(deltas,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['select','evaluate']);a=p.parse_args();select() if a.action=='select' else evaluate()

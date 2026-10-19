"""Same-fold ensemble distillation with a matched hard-label student control."""
import argparse,hashlib,json,pickle,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from models.capacity_large import CapacityLarge
from models.stclassifier import PseTae
from dataset import PixelSetData
from learning.weight_init import weight_init
from scripts.train_tgt_scratch import PatchInputs
from scripts.cnn_correction_probes import batches,to_device
from scripts.tgt_correction_probes import predict
from scripts.train_pixel_temporal import runtime,EpochSampler
from scripts.frozen_trajectory_probes import write_json,metrics,partitions
from scripts.evaluate_record_ensemble import save_predictions

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def softened_target(p,temperature):
 return torch.softmax(torch.log(p.clamp_min(1e-12))/temperature,dim=-1).detach()
def loss_terms(logits,labels,target,alpha,temperature):
 assert not target.requires_grad
 ce=F.cross_entropy(logits,labels)
 kl=F.kl_div(F.log_softmax(logits/temperature,dim=-1),target,reduction='batchmean')*(temperature**2)
 return (1-alpha)*ce+alpha*kl,ce,kl
class NativeBaseline(PseTae):
 """Original model; flatten the call interface for the existing batch helpers."""
 def forward(self,pixels,mask,geometry):
  return super().forward(((pixels,mask),(geometry,None)))
class NativeInputs(PixelSetData):
 def __init__(self,a):
  with open(Path(a.native)/'S2-2017-T31TFM-meanstd.pkl','rb') as f:norm=pickle.load(f)
  super().__init__(a.native,labels='label_44class',npixel=64,sub_classes=[1,3,4,5,6,8,9,12,13,14,16,18,19,23,28,31,33,34,36,39],norm=norm,extra_feature='geomfeat',jitter=None)
  self.seed=a.seed;self.pids=np.asarray(self.pid);self.y=np.asarray(self.target);self.days=np.asarray(self.date_positions);self.splits=partitions(len(self.y),a.fold)
  with np.load(Path(a.cache)/f'fold_{a.fold}_features.npz') as f:
   assert np.array_equal(self.pids,f['pids']) and np.array_equal(self.y,f['labels']) and np.array_equal(self.days,f['days'])
 def __getitem__(self,key):
  training=isinstance(key,tuple);i,epoch=key if training else (int(key),None)
  seed=int(np.random.SeedSequence([self.seed,int(epoch),int(self.pid[i])]).generate_state(1)[0]) if training else (20261013+int(self.pid[i]))%2**32
  np.random.seed(seed);self.jitter=(.01,.05) if training else None
  ((pixels,mask),geometry),y=super().__getitem__(int(i))
  return dict(pixels=pixels,mask=mask,geometry=geometry),int(y),int(i)

def state_sha(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():
  h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
class TrainingTargets(Dataset):
 def __init__(self,base,ids,p,temperature):
  assert np.array_equal(ids,base.splits['train']),'Teacher targets must match exactly the original TRAIN order'
  assert p.shape==(len(ids),20) and np.isfinite(p).all() and (p>=0).all()
  np.testing.assert_allclose(p.sum(1),1,atol=1e-7,rtol=0)
  self.base=base;self.row=np.full(len(base.y),-1,dtype=np.int64);self.row[ids]=np.arange(len(ids))
  self.q=softened_target(torch.as_tensor(p,dtype=torch.float64),temperature).float().numpy()
 def __len__(self):return len(self.base.y)
 def __getitem__(self,key):
  i=key[0] if isinstance(key,tuple) else key
  assert self.row[i]>=0,'Validation/test parcel requested from distillation training data'
  batch,y,ix=self.base[key];return batch,y,ix,self.q[self.row[i]]
def setup(a):
 root=Path(a.output)/f'Fold_{a.fold}';exp=json.loads((root/'experiment.json').read_text());assert exp['fold']==a.fold and 1<=a.fold<=5
 if exp['student'].get('initialization')=='fresh_random':
  assert a.seed==exp['training']['seed']==1
  ds=NativeInputs(a);audit=json.loads((root/'teacher_audit.json').read_text());assert audit['fold']==a.fold
  assert audit['split_sha256']=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in ds.splits.items()}
  assert audit['pid_sha256']==hashlib.sha256(','.join(ds.pids).encode()).hexdigest()
  model=NativeBaseline(positions=list(ds.days));model.apply(weight_init)
  reference=torch.load(exp['student']['reference_path'],map_location='cpu',weights_only=False)
  assert set(model.state_dict())==set(reference['state_dict'])
  assert all(v.shape==reference['state_dict'][k].shape for k,v in model.state_dict().items())
  assert sha(exp['student']['reference_path'])==audit['native_source_sha256']
  assert state_sha(model)!=state_sha_from_checkpoint(reference,ds.days)
  return root,exp,ds,None,model.to(a.device)
 source=Path(exp['student']['initialization_path']);assert sha(source)==exp['student']['initialization_sha256']
 ds=PatchInputs(a);ck=torch.load(source,map_location='cpu',weights_only=False);assert ck['epoch']==45 and ck['protocol']['fold']==1
 assert ck['protocol']['split_hashes']=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in ds.splits.items()}
 assert ck['protocol']['pid_sha256']==hashlib.sha256(','.join(ds.pids).encode()).hexdigest()
 for k,v in [('spectral_mean',ds.mean),('spectral_std',ds.std),('days',ds.days),('geometry_mean',ds.geom_mean),('geometry_std',ds.geom_std)]:np.testing.assert_array_equal(ck['protocol'][k],v)
 model=CapacityLarge(ds.days,ds.mean,ds.std);model.load_state_dict(ck['state_dict']);assert model.groups is None
 return root,exp,ds,ck,model.to(a.device)
def state_sha_from_checkpoint(ck,days):
 # Isolate construction so this audit cannot change the student's RNG stream.
 with torch.random.fork_rng(devices=[]):
  reference=NativeBaseline(positions=list(days));reference.load_state_dict(ck['state_dict']);return state_sha(reference)
def train(a):
 runtime(1);root,exp,ds,source,model=setup(a);config=exp['training'];alpha=config['alpha'] if a.variant=='distilled' else 0.;T=config['temperature'];folder=root/a.variant;folder.mkdir(exist_ok=True)
 complete=json.loads((root/'teacher_complete.json').read_text())
 if 'teacher_cache' in exp:
  assert complete['experiment_sha256']==sha(root/'teacher_source_experiment.json')==exp['teacher_cache']['source_experiment_sha256']
  assert sha(root/'teacher_complete.json')==exp['teacher_cache']['manifest_sha256']
  assert json.loads((root/'teacher_source_experiment.json').read_text())['members']==exp['members']
 else:assert complete['experiment_sha256']==sha(root/'experiment.json')
 assert complete['teacher_train_sha256']==sha(root/'teacher_train.npz')
 assert complete['split_sha256']=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in ds.splits.items()}
 with np.load(root/'teacher_train.npz') as f:training=TrainingTargets(ds,f['indices'],f['probabilities'],T)
 protocol=dict(fold=a.fold,variant=a.variant,experiment_sha256=sha(root/'experiment.json'),teacher_sha256=sha(root/'teacher_train.npz'),initial_state_sha256=state_sha(model),alpha=alpha,temperature=T,epochs=config['epochs'],parameters=sum(p.numel() for p in model.parameters()),split_sha256=complete['split_sha256'],loss=config['loss'] if alpha else 'CE(y,z)',optimizer=f"Adam lr{config['lr']}; no weight decay/scheduler; clip5",initialization=exp['student'].get('initialization','Large-capacity fold1 epoch45'),architecture=exp['student']['architecture'])
 if (folder/'protocol.json').exists():assert json.loads((folder/'protocol.json').read_text())==protocol
 else:write_json(folder/'protocol.json',protocol)
 if (folder/'done.json').exists():return
 a.batch_size=config['batch_size'];sampler=EpochSampler(ds.splits['train'],1);tr=batches(training,sampler,a);va=batches(ds,ds.splits['val'].tolist(),a)
 optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']);history=[];start=1;best_epoch=0;best=-1.
 def status(stage,**kw):write_json(folder/'status.json',dict(stage=stage,variant=a.variant,updated_unix=time.time(),**kw))
 def best_checkpoint(epoch,m,z,ids):
  torch.save(dict(state_dict=model.state_dict(),epoch=epoch,validation=m,protocol=protocol),folder/'best.tmp.pt');(folder/'best.tmp.pt').replace(folder/'best.pt');save_predictions(folder/'validation_predictions.npz',ids,ds.y[ids],z)
 def last_checkpoint(epoch):
  torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,history=history,protocol=protocol,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else []),folder/'last.tmp.pt');(folder/'last.tmp.pt').replace(folder/'last.pt')
 if (folder/'last.pt').exists():
  last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);assert last['protocol']==protocol;model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);history=last['history'];start=last['epoch']+1;best=last['best'];best_epoch=last['best_epoch'];torch.set_rng_state(last['torch_rng'])
  if a.device.startswith('cuda'):torch.cuda.set_rng_state_all(last['cuda_rng'])
 else:
  status('initial_validation');z,y,ids=predict(model,va,a.device);assert np.array_equal(ids,ds.splits['val']);m=metrics(y,z.argmax(1))
  if source is not None:
   for key in ['miou','oa','mean_recall']:assert abs(m[key]-source['validation'][key])<1e-12
  best=m['miou'];history=[dict(epoch=0,validation=m,initialization=protocol['initialization'])];best_checkpoint(0,m,z,ids);last_checkpoint(0);write_json(folder/'history.json',history)
 for epoch in range(start,config['epochs']+1):
  sampler.epoch=epoch;model.train();tick=time.time();sums=np.zeros(3);count=0
  for step,(batch,y,ids,q) in enumerate(tr):
   optimizer.zero_grad(set_to_none=True);z=model(**to_device(batch,a.device));loss,ce,kl=loss_terms(z,y.to(a.device),q.to(a.device),alpha,T);assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
   sums+=np.array([loss.item(),ce.item(),kl.item()])*len(y);count+=len(y)
   if step%100==0:
    record=dict(epoch=epoch,step=step,steps=len(tr),train_loss=loss.item(),ce=ce.item(),kl_T2=kl.item(),gradient_norm=float(grad),seconds=time.time()-tick,best_epoch=best_epoch,best_validation_miou=best);status('training',**record);print(json.dumps(record),flush=True)
  status('validation',epoch=epoch,best_epoch=best_epoch,best_validation_miou=best)
  z,y,ids=predict(model,va,a.device);assert np.array_equal(ids,ds.splits['val']) and np.array_equal(y,ds.y[ids]);m=metrics(y,z.argmax(1));row=dict(epoch=epoch,train_loss=float(sums[0]/count),ce=float(sums[1]/count),kl_T2=float(sums[2]/count),validation=m,seconds=time.time()-tick);history.append(row)
  if m['miou']>best:best=m['miou'];best_epoch=epoch;best_checkpoint(epoch,m,z,ids)
  last_checkpoint(epoch);write_json(folder/'history.json',history);status('epoch_complete',epoch=epoch,best_epoch=best_epoch,best_validation_miou=best);print(json.dumps(dict(epoch=epoch,validation_miou=m['miou'],best_epoch=best_epoch,best_miou=best,seconds=row['seconds'])),flush=True)
 write_json(folder/'done.json',dict(epochs=config['epochs'],best_epoch=best_epoch,best_validation_miou=best,completed_unix=time.time()));status('complete',epoch=config['epochs'],best_epoch=best_epoch,best_validation_miou=best)
def evaluate(a):
 runtime(1);root,exp,ds,source,model=setup(a)
 assert all((root/v/'done.json').exists() for v in ['distilled','hard_label'])
 selected={v:torch.load(root/v/'best.pt',map_location='cpu',weights_only=False) for v in ['distilled','hard_label']}
 write_json(root/'test_selection.json',dict(criterion='Validation mIoU only; both student fits complete before test',best_epochs={n:c['epoch'] for n,c in selected.items()},checkpoint_sha256={n:sha(root/n/'best.pt') for n in selected}))
 ids=ds.splits['test'];loader=batches(ds,ids.tolist(),a);scores={};preds={}
 reference_name='reference_baseline' if source is None else 'initial_large_capacity'
 if source is None:source=torch.load(exp['student']['reference_path'],map_location='cpu',weights_only=False)
 for name,ck in [(reference_name,source)]+list(selected.items()):
  model.load_state_dict(ck['state_dict']);z,y,ix=predict(model,loader,a.device);assert np.array_equal(ix,ids) and np.array_equal(y,ds.y[ids]);scores[name]=metrics(y,z.argmax(1));preds[name]=z.argmax(1);save_predictions(root/f'{name}_test.npz',ix,y,z)
 with np.load(root/'reference_nine_test.npz') as f:assert np.array_equal(ids,f['indices']) and np.array_equal(ds.y[ids],f['labels']);preds['teacher']=f['probabilities'].argmax(1)
 scores['teacher']=metrics(ds.y[ids],preds['teacher'])
 if a.fold==1:assert abs(scores['teacher']['miou']-.5493308503001464)<1e-12
 rng=np.random.RandomState(20261019);draws={n:[] for n in ['hard_label',reference_name,'teacher']};y=ds.y[ids]
 for _ in range(1000):
  sample=rng.randint(0,len(ids),len(ids));student=metrics(y[sample],preds['distilled'][sample])['miou']
  for n in draws:draws[n].append(student-metrics(y[sample],preds[n][sample])['miou'])
 write_json(root/'report.json',dict(fold=a.fold,n_test=len(ids),metrics=scores,selection={n:c['epoch'] for n,c in selected.items()},paired_distilled_minus={n:dict(delta=scores['distilled']['miou']-scores[n]['miou'],ci95=np.quantile(d,[.025,.975]).tolist()) for n,d in draws.items()},caveat='Teacher recipe selected during prior fold1 exploration; overlapping folds and validation sets. Paired intervals conditional on fitted models; not seed/fold/spatial/selection uncertainty.'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate']);p.add_argument('--variant',choices=['distilled','hard_label'],default='distilled');p.add_argument('--output',default='results/distillation');p.add_argument('--cache',default='results/pair_probes');p.add_argument('--native',default='data/S2-2017-T31TFM-PixelSet');p.add_argument('--patch',default='data/s2-2017-IGARSS-NNI-NPY');p.add_argument('--fold',type=int,default=1,choices=range(1,6));p.add_argument('--seed',type=int,default=1);p.add_argument('--workers',type=int,default=2);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--device',default='cuda');a=p.parse_args();train(a) if a.action=='train' else evaluate(a)

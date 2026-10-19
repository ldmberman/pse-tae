"""Check KL direction/temperature, split exclusion, real-data gradients and exact resume."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from scripts.distill_pse_tae import loss_terms,softened_target,TrainingTargets,setup,state_sha,NativeBaseline
from scripts.cnn_correction_probes import batches,to_device
from scripts.train_pixel_temporal import runtime
from scripts.frozen_trajectory_probes import write_json

def main(a):
 runtime(1)
 z=torch.tensor([[1.,2.,-1.],[.4,-.3,.2]],requires_grad=True);y=torch.tensor([1,0]);p=torch.softmax(z.detach(),dim=-1);q=softened_target(p,2)
 loss,ce,kl=loss_terms(z,y,q,.5,2);assert abs(kl.item())<1e-6
 zero,_,_=loss_terms(z,y,q,0,2);torch.testing.assert_close(zero,F.cross_entropy(z,y))
 grads=torch.autograd.grad(zero,z,retain_graph=True)[0];torch.testing.assert_close(grads,torch.autograd.grad(F.cross_entropy(z,y),z)[0])
 assert not q.requires_grad;torch.testing.assert_close(q,torch.softmax(z.detach()/2,dim=-1))
 root,exp,ds,source,model=setup(a);tr=ds.splits['train'];uniform=np.ones((len(tr),20))/20;wrapped=TrainingTargets(ds,tr,uniform,2)
 for bad in [ds.splits['val'][0],ds.splits['test'][0]]:
  try:wrapped[int(bad)]
  except AssertionError:pass
  else:raise AssertionError('Nontraining parcel accepted')
 try:TrainingTargets(ds,tr[::-1],uniform,2)
 except AssertionError:pass
 else:raise AssertionError('Reordered teacher IDs accepted')
 if isinstance(model,NativeBaseline):
  assert model.temporal_encoder is not None and not model.date_kurtosis
  initial_hash=state_sha(model);runtime(1);_,_,other,_,second=setup(a);assert state_sha(second)==initial_hash
  for key in [(int(tr[0]),1),(int(tr[0]),2),int(ds.splits['val'][0])]:
   left=ds[key];right=other[key]
   for k in left[0]:torch.testing.assert_close(left[0][k],right[0][k],atol=0,rtol=0)
  assert not torch.equal(ds[(int(tr[0]),1)][0]['pixels'],ds[(int(tr[0]),2)][0]['pixels'])
  second.load_state_dict(torch.load(exp['student']['reference_path'],map_location='cpu',weights_only=False)['state_dict']);second.eval()
  vx,vy,vi=next(iter(batches(ds,ds.splits['val'][:64].tolist(),a)))
  with torch.no_grad():actual=second(**to_device(vx,a.device)).cpu().numpy()
  with np.load(Path(a.cache)/f'fold_{a.fold}_features.npz') as f:expected=f['logits'][vi.numpy()]
  np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4);assert np.array_equal(actual.argmax(1),expected.argmax(1))
  del second,other
 else:assert model.groups is None and model.base.temporal_encoder is not None
 batch,y,ids,q=next(iter(batches(wrapped,tr[:128].tolist(),a)));batch=to_device(batch,a.device);y=y.to(a.device);q=q.to(a.device)
 if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
 model.train();optimizer=torch.optim.Adam(model.parameters(),lr=exp['training']['lr'])
 def step():
  optimizer.zero_grad(set_to_none=True);loss,ce,kl=loss_terms(model(**batch),y,q,.5,2);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();return float(loss.detach())
 step();state=copy.deepcopy(model.state_dict());opt=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [];expected_loss=step();expected=copy.deepcopy(model.state_dict())
 model.load_state_dict(state);optimizer.load_state_dict(opt);torch.set_rng_state(rng)
 if a.device.startswith('cuda'):torch.cuda.set_rng_state_all(cuda)
 actual_loss=step();assert expected_loss==actual_loss
 for name,value in model.state_dict().items():torch.testing.assert_close(value,expected[name],atol=0,rtol=0)
 result=dict(kl_temperature_direction=True,alpha_zero_equals_cross_entropy=True,targets_detached=True,validation_and_test_ids_rejected=True,reordered_train_targets_rejected=True,single_pse_tae_no_cnn_or_group_branch=True,real_batch=128,finite_gradients=True,exact_optimizer_rng_resume=True,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if a.device.startswith('cuda') else None,parameters=sum(p.numel() for p in model.parameters()),experiment_sha256=__import__('hashlib').sha256((root/'experiment.json').read_bytes()).hexdigest())
 if isinstance(model,NativeBaseline):result.update(fresh_original_baseline_shape=True,identical_student_control_initialization=True,initial_state_sha256=initial_hash,matched_resampling_jitter=True,original_baseline_validation_logits_reproduced=True)
 write_json(root/'checks.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default='results/distillation');p.add_argument('--cache',default='results/pair_probes');p.add_argument('--native',default='data/S2-2017-T31TFM-PixelSet');p.add_argument('--patch',default='data/s2-2017-IGARSS-NNI-NPY');p.add_argument('--fold',type=int,default=1);p.add_argument('--seed',type=int,default=1);p.add_argument('--workers',type=int,default=0);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--device',default='cuda');main(p.parse_args())

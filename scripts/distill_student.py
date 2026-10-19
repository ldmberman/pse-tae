"""Fresh baseline-size student or CE control, from same-fold saved teacher targets.

The targets must be the complete original TRAIN partition in its original order.
Validation chooses the checkpoint; this script never reads test predictions.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.distill_pse_tae import NativeInputs,NativeBaseline,TrainingTargets,loss_terms,state_sha
from scripts.train_pixel_temporal import runtime,EpochSampler
from scripts.cnn_correction_probes import batches,to_device,predict
from scripts.frozen_trajectory_probes import metrics
from scripts.evaluate_checkpoint import digest
from learning.weight_init import weight_init

def run(a):
    runtime(a.seed);ds=NativeInputs(a);model=NativeBaseline(positions=list(ds.days));model.apply(weight_init);model.to(a.device)
    alpha=.5 if a.variant=='distilled' else 0.
    with np.load(a.teacher) as f:
        assert int(f['fold'])==a.fold and str(f['split'])=='train' and not bool(f['partial'])
        assert np.array_equal(f['labels'],ds.y[ds.splits['train']])
        training=TrainingTargets(ds,f['indices'],f['probabilities'],2.)
    out=Path(a.output)/f'Fold_{a.fold}'/a.variant;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(fold=a.fold,variant=a.variant,teacher_sha256=digest(a.teacher),initial_state_sha256=state_sha(model),epochs=a.epochs,alpha=alpha,temperature=2.,seed=a.seed,lr=.001,clip=5.,batch_size=a.batch_size)
    sampler=EpochSampler(ds.splits['train'],a.seed);loader=batches(training,sampler,a);val=batches(ds,ds.splits['val'].tolist(),a)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);history=[];start=1;best=-1.;best_epoch=0
    def validate(epoch):
        nonlocal best,best_epoch
        z,y,ids=predict(model,val,a.device);m=metrics(y,z.argmax(1));history.append(dict(epoch=epoch,validation=m))
        if m['miou']>best:
            best=m['miou'];best_epoch=epoch
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,validation=m,protocol=protocol),out/'best.tmp.pt');(out/'best.tmp.pt').replace(out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(dict(epoch=epoch,validation_miou=m['miou'],best_epoch=best_epoch)),flush=True)
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert ck['protocol']==protocol
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);start=ck['epoch']+1;history=ck['history'];best=ck['best'];best_epoch=ck['best_epoch'];torch.set_rng_state(ck['torch_rng'])
        if a.device.startswith('cuda'):torch.cuda.set_rng_state_all(ck['cuda_rng'])
    else:validate(0)
    for epoch in range(start,a.epochs+1):
        sampler.epoch=epoch;model.train()
        for x,y,ids,q in loader:
            optimizer.zero_grad(set_to_none=True);z=model(**to_device(x,a.device));loss,_,_=loss_terms(z,y.to(a.device),q.to(a.device),alpha,2.)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        validate(epoch)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,history=history,best=best,best_epoch=best_epoch,protocol=protocol,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else []),out/'last.tmp.pt');(out/'last.tmp.pt').replace(out/'last.pt')
    (out/'done.json').write_text(json.dumps(dict(epochs=a.epochs,best_epoch=best_epoch)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--teacher',required=True);p.add_argument('--variant',choices=['distilled','hard_label'],default='distilled');p.add_argument('--fold',type=int,choices=range(1,6),required=True);p.add_argument('--cache',default='results/pair_probes');p.add_argument('--native',default='data/S2-2017-T31TFM-PixelSet');p.add_argument('--output',default='results/distillation');p.add_argument('--seed',type=int,default=1);p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--workers',type=int,default=2);p.add_argument('--device',default='cuda');run(p.parse_args())

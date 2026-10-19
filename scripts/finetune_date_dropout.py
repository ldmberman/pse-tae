"""Paired all-date versus date-dropout fine-tuning from one immutable checkpoint."""
import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.date_dropout import DateDropoutCapacity
from scripts.train_tgt_scratch import PatchInputs
from scripts.tgt_correction_probes import batches, to_device, predict
from scripts.frozen_trajectory_probes import metrics, focal_loss, write_json
from scripts.train_pixel_temporal import EpochSampler, runtime

VARIANTS = ['date_dropout', 'all_dates']


def save_predictions(path, indices, labels, logits):
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f, indices=indices, labels=labels, logits=logits)
    path.with_suffix('.tmp').replace(path)


def select_dates(seed, epoch, step, count, probability):
    if probability == 0:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([20261018, seed, epoch, step]))
    ix = np.flatnonzero(rng.random(count) >= probability)
    if len(ix) < 2:
        ix = np.sort(rng.choice(count, size=2, replace=False))
    return ix


def build(args, ds):
    ck = torch.load(Path(args.output)/'starting.pt', map_location='cpu', weights_only=False)
    p = ck['protocol']
    assert p['variant'] == 'capacity' and p['fold'] == args.fold
    assert p['pid_sha256'] == hashlib.sha256(','.join(ds.pids).encode()).hexdigest()
    assert p['split_hashes'] == {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in ds.splits.items()}
    for name, actual in [('spectral_mean',ds.mean),('spectral_std',ds.std),('days',ds.days),
                         ('geometry_mean',ds.geom_mean),('geometry_std',ds.geom_std)]:
        np.testing.assert_allclose(p[name], actual, rtol=0, atol=0)
    model = DateDropoutCapacity(ds.days, ds.mean, ds.std)
    model.load_state_dict(ck['state_dict'], strict=True)
    return model.to(args.device), ck


def train(args):
    runtime(args.seed)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    folder = Path(args.output)/f'Fold_{args.fold}'/args.variant
    folder.mkdir(parents=True, exist_ok=True)
    ds = PatchInputs(args)
    model, source = build(args, ds)
    probability = args.drop_probability if args.variant == 'date_dropout' else 0.
    protocol = dict(vars(args), source=json.loads((Path(args.output)/'source.json').read_text()),
        effective_drop_probability=probability,
        date_sampling='Independent Bernoulli per date, one subset per batch; minimum two retained dates, chronological order. Separate seed/epoch/step RNG.',
        positional_encoding='Select original acquisition-day sinusoid rows; remove omitted dates before PSE, query averaging and attention.',
        optimization='All parameters fine-tuned; fresh Adam in both arms, lr.0001, no weight decay/scheduler/clipping. Original focal gamma1 and internal dropout.',
        comparison='Same starting weights, batch size, parcel order, epoch budget and full-date validation. Epoch0 eligible. Test only after both arms.',
        limitation='Exploratory fold1; source checkpoint chosen using this validation set. Not an independent confirmation.')
    if (folder/'protocol.json').exists():
        assert json.loads((folder/'protocol.json').read_text()) == protocol
    write_json(folder/'protocol.json', protocol)
    if (folder/'done.json').exists():
        return
    tr,va = ds.splits['train'],ds.splits['val']
    if args.smoke:
        tr,va=tr[:2*args.batch_size],va[:2*args.batch_size]
    sampler=EpochSampler(tr,args.seed)
    train_loader,val_loader=batches(ds,sampler,args),batches(ds,va.tolist(),args)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    history,start,best_epoch=[],1,0
    def save_best(epoch,score,logits):
        torch.save(dict(state_dict=model.state_dict(),epoch=epoch,validation=score,protocol=protocol),folder/'best.tmp.pt')
        (folder/'best.tmp.pt').replace(folder/'best.pt')
        save_predictions(folder/'validation_predictions.npz',va,ds.y[va],logits)
    if (folder/'last.pt').exists():
        ck=torch.load(folder/'last.pt',map_location=args.device,weights_only=False)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        history,start,best,best_epoch=ck['history'],ck['epoch']+1,ck['best'],ck['best_epoch']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if args.device=='cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    else:
        print(json.dumps(dict(stage='initial_validation',variant=args.variant,source_epoch=source['epoch'])),flush=True)
        logits,y,ix=predict(model,val_loader,args.device)
        assert np.array_equal(ix,va) and np.array_equal(y,ds.y[va])
        score=metrics(y,logits.argmax(1));best=score['miou']
        if not args.smoke:
            assert abs(best-source['validation']['miou'])<1e-12,(best,source['validation'])
        save_best(0,score,logits)
        history=[dict(epoch=0,validation=score,source_equivalent=True)]
        write_json(folder/'history.json',history)
    print(json.dumps(dict(stage='training',variant=args.variant,start_epoch=start,best_miou=best)),flush=True)
    for epoch in range(start,args.epochs+1):
        sampler.epoch=epoch
        model.train();tick=time.time();loss_sum=count=retained=steps=0
        for step,(batch,y,_) in enumerate(train_loader):
            ix=select_dates(args.seed,epoch,step,len(ds.days),probability)
            inputs=to_device(batch,args.device)
            optimizer.zero_grad(set_to_none=True)
            loss=focal_loss(model(**inputs,date_indices=ix),y.to(args.device))
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            optimizer.step()
            loss_sum+=loss.item()*len(y);count+=len(y);retained+=len(ds.days) if ix is None else len(ix);steps+=1
            if step%100==0:
                print(json.dumps(dict(epoch=epoch,step=step,loss=loss.item(),gradient_norm=float(norm),
                    retained_dates=len(ds.days) if ix is None else len(ix),seconds=time.time()-tick)),flush=True)
        logits,y,ix=predict(model,val_loader,args.device)
        assert np.array_equal(ix,va) and np.array_equal(y,ds.y[va])
        score=metrics(y,logits.argmax(1))
        history.append(dict(epoch=epoch,train_loss=loss_sum/count,mean_retained_dates=retained/steps,
                            validation=score,seconds=time.time()-tick))
        if score['miou']>best:
            best,best_epoch=score['miou'],epoch
            save_best(epoch,score,logits)
        ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,
            epoch=epoch,best=best,best_epoch=best_epoch,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else [])
        torch.save(ck,folder/'last.tmp.pt');(folder/'last.tmp.pt').replace(folder/'last.pt')
        write_json(folder/'history.json',history)
        write_json(folder/'status.json',dict(epoch=epoch,best_epoch=best_epoch,best_miou=best,updated_unix=time.time()))
        print(json.dumps(dict(epoch=epoch,validation=score['miou'],best_miou=best,best_epoch=best_epoch,seconds=time.time()-tick)),flush=True)
    write_json(folder/'done.json',dict(epochs=args.epochs,best_epoch=best_epoch,best_validation_miou=best,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if args.device=='cuda' else None))


def evaluate(args):
    runtime(args.seed)
    root=Path(args.output)/f'Fold_{args.fold}'
    assert all((root/v/'done.json').exists() for v in VARIANTS)
    ck={v:torch.load(root/v/'best.pt',map_location='cpu',weights_only=False) for v in VARIANTS}
    selected=max(VARIANTS,key=lambda v:ck[v]['validation']['miou'])
    write_json(root/'selection.json',dict(selected=selected,criterion='Validation mIoU including epoch0',
        best_epochs={v:c['epoch'] for v,c in ck.items()}))
    ds=PatchInputs(args);test=ds.splits['test']
    if args.smoke:
        test=test[:2*args.batch_size]
    model,_=build(args,ds)
    scores,preds={},{}
    for v in ['starting']+VARIANTS:
        if v!='starting':
            model.load_state_dict(ck[v]['state_dict'])
        logits,y,ix=predict(model,batches(ds,test.tolist(),args),args.device)
        assert np.array_equal(ix,test) and np.array_equal(y,ds.y[test])
        preds[v]=logits.argmax(1);scores[v]=metrics(y,preds[v])
        folder=root/v;folder.mkdir(exist_ok=True)
        save_predictions(folder/'test_predictions.npz',ix,y,logits)
    pairs=[('date_dropout','all_dates'),('date_dropout','starting'),('all_dates','starting')]
    draws={a+'-minus-'+b:[] for a,b in pairs};rng=np.random.RandomState(20261018)
    for _ in range(1000):
        ix=rng.randint(len(y),size=len(y));m={v:metrics(y[ix],p[ix])['miou'] for v,p in preds.items()}
        for a,b in pairs:draws[a+'-minus-'+b].append(m[a]-m[b])
    report=dict(selected=selected,metrics=scores,paired_miou_differences={a+'-minus-'+b:
        dict(delta=scores[a]['miou']-scores[b]['miou'],ci95=np.quantile(draws[a+'-minus-'+b],[.025,.975]).tolist()) for a,b in pairs},
        caveat='Exploratory fold1. Parcel bootstrap conditions on fitted models; does not measure seed/fold/spatial uncertainty.')
    write_json(root/'report.json',report)
    lines=['# Date-dropout fine-tuning','',report['caveat'],'',f'Validation-selected: {selected}.','',
           '| Model | Test OA % | Test mIoU % | Test mean recall % |','|---|---:|---:|---:|']
    for v,s in scores.items():lines.append(f'| {v} | {100*s["oa"]:.3f} | {100*s["miou"]:.3f} | {100*s["mean_recall"]:.3f} |')
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def run(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (root/'starting.pt').exists():
            data=Path(args.source).read_bytes()
            ck=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
            assert ck['protocol']['variant']=='capacity' and ck['protocol']['fold']==args.fold
            write_json(root/'source.json',dict(path=args.source,sha256=hashlib.sha256(data).hexdigest(),
                epoch=ck['epoch'],validation=ck['validation'],captured_unix=time.time()))
            (root/'starting.tmp.pt').write_bytes(data);(root/'starting.tmp.pt').replace(root/'starting.pt')
        record=json.loads((root/'source.json').read_text())
        assert hashlib.sha256((root/'starting.pt').read_bytes()).hexdigest()==record['sha256']
        common=['--output',args.output,'--cache',args.cache,'--patch',args.patch,'--native',args.native,
            '--fold',str(args.fold),'--seed',str(args.seed),'--epochs',str(args.epochs),'--batch-size',str(args.batch_size),
            '--workers',str(args.workers),'--lr',str(args.lr),'--device',args.device,'--drop-probability',str(args.drop_probability)]
        if args.smoke:common+=['--smoke']
        complete=[]
        def status(stage,**kw):write_json(root/'status.json',dict(stage=stage,complete=complete,updated_unix=time.time(),**kw))
        try:
            for v in VARIANTS:
                status('training',active=v)
                with (root/(v+'.log')).open('a') as log:
                    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'train','--variant',v]+common,stdout=log,stderr=subprocess.STDOUT,check=True)
                complete.append(v)
            status('evaluating')
            with (root/'evaluate.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'evaluate']+common,stdout=log,stderr=subprocess.STDOUT,check=True)
            status('complete')
        except Exception as exc:
            status('failed',error=repr(exc));raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','train','evaluate'])
    p.add_argument('--variant',choices=VARIANTS,default='date_dropout')
    p.add_argument('--source',default='results/grouped_scratch/Fold_1/capacity/best.pt')
    p.add_argument('--output',default='results/date_dropout')
    p.add_argument('--cache',default='results/pair_probes')
    p.add_argument('--patch',default='data/s2-2017-IGARSS-NNI-NPY')
    p.add_argument('--native',default='data/S2-2017-T31TFM-PixelSet')
    p.add_argument('--fold',type=int,default=1)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--drop-probability',type=float,default=.2)
    p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    assert 0<=args.drop_probability<1 and args.epochs>=1
    globals()[args.command](args)

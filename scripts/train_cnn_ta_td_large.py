"""From-scratch large CNN + temporal attention + date dropout, fold 1."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.cnn_ta_td_large import CNNTATDLarge
from models.capacity_large import CapacityLarge
from scripts.finetune_date_dropout import select_dates
from models.tgt_scratch import parameter_count
from scripts.train_tgt_scratch import PatchInputs
from scripts.tgt_correction_probes import batches, to_device, predict
from scripts.frozen_trajectory_probes import metrics, focal_loss, write_json
from scripts.train_pixel_temporal import EpochSampler, runtime


def build(ds, variant):
    assert variant == 'cnn_ta_td_large'
    return CNNTATDLarge(ds.days, ds.mean, ds.std)


def train(args):
    runtime(args.seed)
    folder = Path(args.output) / f'Fold_{args.fold}' / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    ds = PatchInputs(args)
    model = build(ds, args.variant).to(args.device)
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    source_files=['models/cnn_ta_td_large.py','scripts/train_cnn_ta_td_large.py','models/capacity_large.py','models/tgt_scratch.py','models/cnn_correction.py','models/tgt_correction.py','models/tae.py','scripts/finetune_date_dropout.py','scripts/train_tgt_scratch.py','scripts/frozen_trajectory_probes.py','scripts/train_pixel_temporal.py']
    source_sha256={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest() for name in source_files}
    protocol = dict(vars(args), source_sha256=source_sha256, trainable_parameters=parameter_count(model),
        inputs='Full 32x32 patches, 10 bands, 4 geometric features; 24 dates before train-time date omission',
        architecture='PSE+TAE384 unchanged; CNN 10-32-64 (two 3x3, LayerNorm/ReLU/mask), cell mean/std128 projected384, shared TAE384 heads4 key32 MLP1536-384-384, cell mean/std768 LayerNorm; concat1152 decoder1152-256-128-20',
        grouping='16 fixed 8x8 pooling cells on CNN features; local 5x5 receptive field crosses cell boundaries; masked sample mean/std normalized by valid counts; not learned cell membership',
        initialization='All learned parameters from scratch; large PSE+TAE encoder initialized identically to capacity_large seed1; new CNN/TAE and expanded decoder randomly initialized',
        loss='Repository focal gamma1 detached probability, unweighted',
        optimization='Adam lr.001, no weight decay or scheduler; no gradient clipping; nonfinite gradients raise',
        augmentation='Drop each date with p0.20, one subset per batch shared by both branches, minimum2; retain original per-date normalization and acquisition-day sinusoid; validation/test all24; native TAE dropout unchanged',
        selection='Best validation mIoU over fixed100 epochs; test only after this model and capacity_large finish. Joint architecture+augmentation experiment, not a parameter-matched ablation',
        missing='Known four missing patches retained, group masks empty, baseline receives one zero normalized dummy; geometry available',
        limitation='Resized grid support is not an independent parcel mask; repeated and blended boundary pixels retained; fold1 exploratory',
        spectral_mean=ds.mean.tolist(), spectral_std=ds.std.tolist(), days=ds.days.tolist(),
        geometry_mean=ds.geom_mean.tolist(), geometry_std=ds.geom_std.tolist(),
        split_sizes={k:len(v) for k,v in ds.splits.items()},
        split_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in ds.splits.items()},
        pid_sha256=hashlib.sha256(','.join(ds.pids).encode()).hexdigest())
    if (folder/'protocol.json').exists():
        assert json.loads((folder/'protocol.json').read_text()) == protocol, 'Protocol changed on resume'
    write_json(folder/'protocol.json', protocol)
    if (folder/'done.json').exists():
        return
    tr, va = ds.splits['train'], ds.splits['val']
    if args.smoke:
        tr, va = tr[:2*args.batch_size], va[:2*args.batch_size]
    sampler = EpochSampler(tr, args.seed)
    train_loader, val_loader = batches(ds, sampler, args), batches(ds, va.tolist(), args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best, best_epoch, start = [], -1., 0, 1
    if (folder/'last.pt').exists():
        ck = torch.load(folder/'last.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        history, best, best_epoch, start = ck['history'], ck['best'], ck['best_epoch'], ck['epoch']+1
        torch.set_rng_state(ck['torch_rng'].cpu())
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    print(json.dumps(dict(variant=args.variant, parameters=parameter_count(model), start_epoch=start)), flush=True)
    stop_requested = [False]
    import signal
    def request_stop(signum, frame):
        stop_requested[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.time()
    for epoch in range(start, args.epochs+1):
        tick, loss_sum, count, retained, steps = time.time(), 0., 0, 0, 0
        sampler.epoch = epoch
        model.train()
        for step, (batch, y, _) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            date_ix = select_dates(args.seed, epoch, step, len(ds.days), args.drop_probability)
            loss = focal_loss(model(**to_device(batch, args.device), date_indices=date_ix), y.to(args.device))
            retained += len(ds.days) if date_ix is None else len(date_ix)
            steps += 1
            assert torch.isfinite(loss), 'Nonfinite training loss'
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()*len(y)
            count += len(y)
            if step % 100 == 0:
                print(json.dumps(dict(epoch=epoch, step=step, loss=loss.item(), gradient_norm=float(norm), retained_dates=len(ds.days) if date_ix is None else len(date_ix), seconds=time.time()-tick)), flush=True)
        logits, labels, ix = predict(model, val_loader, args.device)
        assert np.array_equal(ix, va) and np.array_equal(labels, ds.y[va])
        score = metrics(labels, logits.argmax(1))
        row = dict(epoch=epoch, train_loss=loss_sum/count, validation=score, mean_retained_dates=retained/steps, seconds=time.time()-tick)
        history.append(row)
        if score['miou'] > best:
            best, best_epoch = score['miou'], epoch
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, validation=score, protocol=protocol), folder/'best.tmp.pt')
            os.replace(folder/'best.tmp.pt', folder/'best.pt')
            np.savez_compressed(folder/'validation_predictions.npz', indices=ix, labels=labels, logits=logits)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                     best=best, best_epoch=best_epoch, history=history, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else [], protocol=protocol)
        torch.save(state, folder/'last.tmp.pt')
        os.replace(folder/'last.tmp.pt', folder/'last.pt')
        write_json(folder/'history.json', history)
        write_json(folder/'status.json', dict(epoch=epoch, best_epoch=best_epoch, best_miou=best, updated_unix=time.time()))
        print(json.dumps(dict(epoch=epoch, validation_miou=score['miou'], validation_oa=score['oa'], best_epoch=best_epoch, best_miou=best, seconds=row['seconds'])), flush=True)
        if stop_requested[0]:
            write_json(Path(args.output)/'status.json', dict(stage='paused', epoch=epoch, updated_unix=time.time(), reason='Signal: saved at epoch boundary'))
            return False
    write_json(folder/'done.json', dict(epochs=args.epochs, best_epoch=best_epoch, best_validation_miou=best, elapsed_seconds=time.time()-started, peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if args.device=='cuda' else None))

    return True


def evaluate_pair(args):
    runtime(args.seed)
    ds = PatchInputs(args)
    split = ds.splits['test']
    if args.smoke:
        split = split[:2*args.batch_size]
    scores = {}
    candidates=[Path(args.controls)/f'Fold_{args.fold}'/'capacity_large',Path(args.output)/f'Fold_{args.fold}'/args.variant]
    choices={}
    for folder in candidates:
        ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        choices[folder.name]=dict(epoch=ck['epoch'],validation=ck['validation'],checkpoint_sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest())
    write_json(Path(args.output)/f'Fold_{args.fold}'/'selection.json',dict(criterion='Best validation mIoU',selected=max(choices,key=lambda k:choices[k]['validation']['miou']),models=choices))
    for name, folder, cls in [
        ('capacity_large', Path(args.controls)/f'Fold_{args.fold}'/'capacity_large', CapacityLarge),
        ('cnn_ta_td_large', Path(args.output)/f'Fold_{args.fold}'/args.variant, CNNTATDLarge)]:
        ck = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        assert ck['protocol']['pid_sha256']==hashlib.sha256(','.join(ds.pids).encode()).hexdigest()
        for key,actual in [('spectral_mean',ds.mean),('spectral_std',ds.std),('days',ds.days),('geometry_mean',ds.geom_mean),('geometry_std',ds.geom_std)]:
            np.testing.assert_allclose(ck['protocol'][key],actual,rtol=0,atol=0)
        assert ck['protocol']['split_hashes'] == {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in ds.splits.items()}
        model = cls(ds.days, ds.mean, ds.std).to(args.device)
        model.load_state_dict(ck['state_dict'])
        logits, labels, indices = predict(model, batches(ds, split.tolist(), args), args.device)
        assert np.array_equal(indices, split) and np.array_equal(labels, ds.y[split])
        np.savez_compressed(folder/'test_predictions_cnn_comparison.npz', indices=indices, labels=labels, logits=logits)
        scores[name] = dict(epoch=ck['epoch'], validation=ck['validation'], test=metrics(labels, logits.argmax(1)))
        del model
    write_json(Path(args.output)/f'Fold_{args.fold}'/'report.json', dict(models=scores,
        caveat='Exploratory fold1 seed1; joint CNN architecture, extra capacity and date dropout. No component attribution. Selection uses validation only.'))


def run(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (root/'runner.pid').write_text(str(os.getpid()))
        def status(stage, **kw):
            write_json(root/'status.json',dict(stage=stage,updated_unix=time.time(),**kw))
        try:
            status('training')
            finished=train(args)
            if finished is False:return
            if args.smoke:
                status('complete_smoke');return
            status('waiting_for_capacity_large')
            while not (Path(args.controls)/f'Fold_{args.fold}'/'capacity_large/done.json').exists():
                time.sleep(30)
            status('evaluating')
            evaluate_pair(args)
            status('complete')
        except Exception as exc:
            status('failed',error=repr(exc));raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','run','evaluate_pair'])
    p.add_argument('--variant',default='cnn_ta_td_large',choices=['cnn_ta_td_large'])
    p.add_argument('--cache',default='results/pair_probes')
    p.add_argument('--patch',default='data/s2-2017-IGARSS-NNI-NPY')
    p.add_argument('--native',default='data/S2-2017-T31TFM-PixelSet')
    p.add_argument('--output',default='results/cnn_ta_td_large')
    p.add_argument('--controls',default='results/capacity_large')
    p.add_argument('--fold',type=int,default=1)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--drop-probability',type=float,default=.2)
    p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    globals()[args.command](args)

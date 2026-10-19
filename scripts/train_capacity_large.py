"""Fourth, larger capacity run; same training loop and data as the three-run study."""
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
from models.capacity_large import CapacityLarge
from models.tgt_scratch import parameter_count
from scripts.train_tgt_scratch import PatchInputs
from scripts.tgt_correction_probes import batches, to_device, predict
from scripts.frozen_trajectory_probes import metrics, focal_loss, write_json
from scripts.train_pixel_temporal import EpochSampler, runtime


def build(ds, variant):
    assert variant == 'capacity_large'
    return CapacityLarge(ds.days, ds.mean, ds.std)


def train(args):
    runtime(args.seed)
    folder = Path(args.output) / f'Fold_{args.fold}' / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    ds = PatchInputs(args)
    model = build(ds, args.variant).to(args.device)
    protocol = dict(vars(args), trainable_parameters=parameter_count(model),
        inputs='Identical full 32x32 patches, all 24 dates and 10 bands; four native geometric features',
        architecture='PSE 10-32-64, pooled132-384; TAE width384 heads4 key32 MLP1536-384-384; decoder384-256-128-20',
        grouping='All valid patch pixels pooled together, identical to original capacity control; no TGT branch',
        initialization='All learned parameters from scratch; same encoder initialization procedure as capacity control; decoder seeded independently',
        loss='Repository focal gamma1 detached probability, unweighted',
        optimization='Adam lr.001, no weight decay or scheduler; no gradient clipping; nonfinite gradients raise',
        augmentation='No pixel subsampling, jitter, or temporal dropout; native TAE internal dropout unchanged',
        selection='Best validation mIoU over fixed 100 epochs; test after training and after original comparison report is available',
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
    started = time.time()
    for epoch in range(start, args.epochs+1):
        tick, loss_sum, count = time.time(), 0., 0
        sampler.epoch = epoch
        model.train()
        for step, (batch, y, _) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss = focal_loss(model(**to_device(batch, args.device)), y.to(args.device))
            assert torch.isfinite(loss), 'Nonfinite training loss'
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()*len(y)
            count += len(y)
            if step % 100 == 0:
                print(json.dumps(dict(epoch=epoch, step=step, loss=loss.item(), gradient_norm=float(norm), seconds=time.time()-tick)), flush=True)
        logits, labels, ix = predict(model, val_loader, args.device)
        assert np.array_equal(ix, va) and np.array_equal(labels, ds.y[va])
        score = metrics(labels, logits.argmax(1))
        row = dict(epoch=epoch, train_loss=loss_sum/count, validation=score, seconds=time.time()-tick)
        history.append(row)
        if score['miou'] > best:
            best, best_epoch = score['miou'], epoch
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, validation=score, protocol=protocol), folder/'best.tmp.pt')
            os.replace(folder/'best.tmp.pt', folder/'best.pt')
            np.savez_compressed(folder/'validation_predictions.npz', indices=ix, labels=labels, logits=logits)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                     best=best, best_epoch=best_epoch, history=history, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else [])
        torch.save(state, folder/'last.tmp.pt')
        os.replace(folder/'last.tmp.pt', folder/'last.pt')
        write_json(folder/'history.json', history)
        write_json(folder/'status.json', dict(epoch=epoch, best_epoch=best_epoch, best_miou=best, updated_unix=time.time()))
        print(json.dumps(dict(epoch=epoch, validation_miou=score['miou'], validation_oa=score['oa'], best_epoch=best_epoch, best_miou=best, seconds=row['seconds'])), flush=True)
    write_json(folder/'done.json', dict(epochs=args.epochs, best_epoch=best_epoch, best_validation_miou=best, elapsed_seconds=time.time()-started))


def evaluate(args):
    runtime(args.seed)
    root = Path(args.output) / f'Fold_{args.fold}'
    folder = root / args.variant
    original = Path(args.controls) / f'Fold_{args.fold}'
    assert (folder/'done.json').exists()
    prior = json.loads((original/'report.json').read_text())
    ck = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    # Record validation selection before reading any additional test predictions.
    best_scores = {v:json.loads((original/v/'done.json').read_text())['best_validation_miou']
                   for v in ['baseline', 'capacity', 'grouped']}
    best_scores[args.variant] = ck['validation']['miou']
    selection = dict(selected=max(best_scores, key=best_scores.get),
                     criterion='validation mIoU', best_validation_miou=best_scores)
    write_json(root/'selection.json', selection)
    ds = PatchInputs(args)
    model = build(ds, args.variant).to(args.device)
    model.load_state_dict(ck['state_dict'])
    test = ds.splits['test']
    logits, y, ix = predict(model, batches(ds, test.tolist(), args), args.device)
    assert np.array_equal(ix, test) and np.array_equal(y, ds.y[test])
    np.savez_compressed(folder/'test_predictions.npz', indices=ix, labels=y, logits=logits)
    predictions = {args.variant:logits.argmax(1)}
    scores = {args.variant:metrics(y, predictions[args.variant])}
    for v in ['baseline', 'capacity', 'grouped']:
        with np.load(original/v/'test_predictions.npz') as f:
            assert np.array_equal(f['indices'], ix) and np.array_equal(f['labels'], y)
            predictions[v] = f['logits'].argmax(1)
        scores[v] = metrics(y, predictions[v])
        assert abs(scores[v]['miou'] - prior['metrics'][v]['miou']) < 1e-12
    rng = np.random.RandomState(20261017)
    differences = {v:[] for v in ['baseline', 'capacity', 'grouped']}
    for _ in range(1000):
        sample = rng.randint(len(y), size=len(y))
        new = metrics(y[sample], predictions[args.variant][sample])['miou']
        for v in differences:
            differences[v].append(new - metrics(y[sample], predictions[v][sample])['miou'])
    report = dict(selection=selection, metrics=scores, paired_miou_differences={
        args.variant+'-minus-'+v:dict(delta=scores[args.variant]['miou']-scores[v]['miou'],
                                     ci95=np.quantile(values, [.025,.975]).tolist())
        for v,values in differences.items()},
        caveat='Exploratory fold1/seed1; larger model chosen after inspecting earlier validation curves. Parcel bootstrap conditions on fitted models; no seed/fold uncertainty.')
    write_json(root/'report.json', report)
    print(json.dumps(report), flush=True)


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root/'runner.pid').write_text(str(os.getpid()))
        def status(stage, **extra):
            write_json(root/'status.json', dict(stage=stage, updated_unix=time.time(), **extra))
        try:
            status('training')
            train(args)
            status('waiting_for_controls')
            controls = Path(args.controls)
            while not (controls/f'Fold_{args.fold}'/'report.json').exists():
                state = json.loads((controls/'status.json').read_text())
                if state['stage'] in {'failed', 'paused'}:
                    raise RuntimeError('Original comparison is '+state['stage']+'; large-model training is saved, evaluation can be resumed.')
                time.sleep(30)
            status('evaluating')
            evaluate(args)
            status('complete')
        except Exception as exc:
            status('failed', error=repr(exc))
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train', 'evaluate', 'run'])
    p.add_argument('--variant', choices=['capacity_large'], default='capacity_large')
    p.add_argument('--cache', default='results/pair_probes')
    p.add_argument('--patch', default='data/s2-2017-IGARSS-NNI-NPY')
    p.add_argument('--native', default='data/S2-2017-T31TFM-PixelSet')
    p.add_argument('--output', default='results/capacity_large')
    p.add_argument('--controls', default='results/grouped_scratch')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    globals()[args.command](args)

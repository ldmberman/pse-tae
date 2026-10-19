"""Fold comparison with identical full patch inputs and no frozen learned inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import Dataset
from models.tgt_scratch import GroupedScratch, VARIANTS, parameter_count
from scripts.tgt_correction_probes import group_patch, batches, to_device, predict
from scripts.frozen_trajectory_probes import (NAMES, MISSING_PATCH_PIDS, partitions,
                                             metrics, focal_loss, write_json)
from scripts.train_pixel_temporal import EpochSampler, runtime


class PatchInputs(Dataset):
    def __init__(self, args):
        self.args = args
        # Metadata only. Never load cached embeddings/logits into training.
        with np.load(Path(args.cache) / f'fold_{args.fold}_features.npz') as f:
            self.pids, self.y, self.days = f['pids'], f['labels'], f['days']
        self.splits = partitions(len(self.y), args.fold)
        with open(Path(args.native) / 'S2-2017-T31TFM-meanstd.pkl', 'rb') as f:
            self.mean, self.std = [np.asarray(v, np.float32) for v in pickle.load(f)]
        assert self.mean.shape == self.std.shape == (24, 10) and (self.std > 0).all()
        geom = json.loads((Path(args.native) / 'META/geomfeat.json').read_text())
        g = np.array([geom[p] for p in self.pids], np.float32)
        tr = self.splits['train']
        self.geom_mean, self.geom_std = g[tr].mean(0), g[tr].std(0).clip(1e-6)
        self.geometry = (g - self.geom_mean) / self.geom_std
        assert self.geometry.shape == (len(self.y), 4) and np.isfinite(self.geometry).all()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, key):
        i = key[0] if isinstance(key, tuple) else key
        path = Path(self.args.patch) / 'DATA' / f'{self.pids[i]}.npy'
        if not path.exists() and self.pids[i] in MISSING_PATCH_PIDS:
            raw, mask = np.zeros((16, 64, 24, 10), np.uint16), np.zeros((16, 64), bool)
        else:
            raw, mask = group_patch(np.load(path))
        return dict(raw=raw.astype(np.float32), mask=mask, geometry=self.geometry[i]), int(self.y[i]), int(i)


def build(ds, variant):
    return GroupedScratch(ds.days, ds.mean, ds.std, variant)


def train(args):
    runtime(args.seed)
    folder = Path(args.output) / f'Fold_{args.fold}' / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    ds = PatchInputs(args)
    model = build(ds, args.variant).to(args.device)
    protocol = dict(vars(args), trainable_parameters=parameter_count(model),
        inputs='Identical full 32x32 patches, all 24 dates and 10 bands; four native geometric features',
        baseline='Original PSE 10-32-64, mean/sample std plus geometry, 132-128; TAE heads4 key32 MLP512-128-128; decoder128-64-32-20',
        capacity='PSE 10-32-64, pooled132-204; TAE width204 heads4 key32 MLP816-204-204; decoder204-128-64-20',
        grouped='Baseline PSE+TAE128 plus independent spectral10-32-64/group mean-std128/TAE128/group mean-std256/LayerNorm256; decoder384-128-64-20',
        grouping='16 fixed 8x8 grid cells, membership fixed across dates; no spatial attention or coordinates',
        initialization='All learned parameters from scratch; shared baseline/grouped encoders identically initialized; no zero-initialized correction',
        loss='Repository focal gamma1 detached probability, unweighted',
        optimization='Adam lr.001, no weight decay or scheduler; no gradient clipping; nonfinite gradients raise',
        augmentation='No pixel subsampling, jitter, or temporal dropout; native TAE internal dropout unchanged',
        selection='Best validation mIoU over fixed 100-epoch budget (unless explicitly overridden); test only after all fits',
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
    root = Path(args.output)/f'Fold_{args.fold}'
    assert all((root/v/'done.json').exists() for v in VARIANTS)
    ck = {v:torch.load(root/v/'best.pt', map_location='cpu', weights_only=False) for v in VARIANTS}
    selected = max(VARIANTS, key=lambda v:ck[v]['validation']['miou'])
    selection = dict(selected=selected, criterion='validation mIoU', best_epochs={v:c['epoch'] for v,c in ck.items()})
    write_json(root/'selection.json', selection)
    ds = PatchInputs(args)
    test = ds.splits['test'][:2*args.batch_size] if args.smoke else ds.splits['test']
    scores, predictions = {}, {}
    for v in VARIANTS:
        model = build(ds, v).to(args.device)
        model.load_state_dict(ck[v]['state_dict'])
        logits, y, ix = predict(model, batches(ds, test.tolist(), args), args.device)
        assert np.array_equal(ix, test) and np.array_equal(y, ds.y[test])
        predictions[v] = logits.argmax(1)
        scores[v] = metrics(y, predictions[v])
        np.savez_compressed(root/v/'test_predictions.npz', indices=ix, labels=y, logits=logits)
        del model
    # Historical native-pixel model: contextual reference, not the matched control.
    with np.load(Path(args.cache)/f'fold_{args.fold}_features.npz') as f:
        predictions['historical_native_baseline'] = f['logits'][test].argmax(1)
    scores['historical_native_baseline'] = metrics(y, predictions['historical_native_baseline'])
    pairs = [('grouped','baseline'),('grouped','capacity'),('capacity','baseline'),('grouped','historical_native_baseline')]
    deltas = {f'{a}-minus-{b}':[] for a,b in pairs}
    rng = np.random.RandomState(20261017)
    for _ in range(1000):
        ix = rng.randint(len(y),size=len(y))
        m = {v:metrics(y[ix],p[ix])['miou'] for v,p in predictions.items()}
        for a,b in pairs:
            deltas[f'{a}-minus-{b}'].append(m[a]-m[b])
    report = dict(selection=selection, metrics=scores, paired_miou_differences={
        f'{a}-minus-{b}':dict(delta=scores[a]['miou']-scores[b]['miou'],ci95=np.quantile(deltas[f'{a}-minus-{b}'],[.025,.975]).tolist()) for a,b in pairs},
        caveat='Exploratory fold1/seed1. Parcel bootstrap conditions on fitted models and does not measure seed/fold variability. Historical model uses different pixel inputs.')
    write_json(root/'report.json', report)
    rows=''.join(f'<tr><td>{v}</td><td>{100*s["oa"]:.3f}</td><td>{100*s["mean_recall"]:.3f}</td><td>{100*s["miou"]:.3f}</td></tr>' for v,s in scores.items())
    perclass=''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{100*s["iou"][i]:.2f}</td>' for s in scores.values())+'</tr>' for i,name in enumerate(NAMES))
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Grouped scratch comparison</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}td,th{padding:8px;text-align:left}tr:nth-child(even){background:#eee}</style>'
        f'<h1>Matched patch-input models trained from scratch</h1><p>Validation-selected: {selected}. {report["caveat"]}</p>'
        '<table><tr><th>Model</th><th>Accuracy %</th><th>Mean recall %</th><th>mIoU %</th></tr>'+rows+'</table><h2>Per-class IoU %</h2><table><tr><th>Class</th>'+''.join(f'<th>{v}</th>' for v in scores)+'</tr>'+perclass+'</table>')
    print(json.dumps({v:{k:s[k] for k in ['oa','miou','mean_recall']} for v,s in scores.items()}),flush=True)


def run(args):
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    (root/'runner.pid').write_text(str(os.getpid()))
    common=['--output',args.output,'--cache',args.cache,'--patch',args.patch,'--native',args.native,
            '--fold',str(args.fold),'--seed',str(args.seed),'--epochs',str(args.epochs),
            '--batch-size',str(args.batch_size),'--workers',str(args.workers),'--lr',str(args.lr),'--device',args.device]+(['--smoke'] if args.smoke else [])
    command=[sys.executable,'-u',str(Path(__file__).resolve())]
    complete=[]
    def fit(v):
        with (root/(v+'.log')).open('a') as log:
            subprocess.run(command+['train','--variant',v]+common,stdout=log,stderr=subprocess.STDOUT,check=True)
        return v
    try:
        # Grouped and baseline start first, for the earliest direct comparison.
        write_json(root/'status.json',dict(stage='training',complete=complete,updated_unix=time.time()))
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for f in as_completed([pool.submit(fit,v) for v in ['baseline','grouped','capacity']]):
                complete.append(f.result())
                write_json(root/'status.json',dict(stage='training',complete=complete,updated_unix=time.time()))
        write_json(root/'status.json',dict(stage='evaluating',complete=complete,updated_unix=time.time()))
        with (root/'evaluate.log').open('a') as log:
            subprocess.run(command+['evaluate']+common,stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(root/'status.json',dict(stage='complete',complete=complete,updated_unix=time.time()))
    except Exception as exc:
        write_json(root/'status.json',dict(stage='failed',complete=complete,error=repr(exc),updated_unix=time.time()))
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','evaluate','run'])
    p.add_argument('--variant',choices=VARIANTS,default='grouped')
    p.add_argument('--cache',default='results/pair_probes')
    p.add_argument('--patch',default='data/s2-2017-IGARSS-NNI-NPY')
    p.add_argument('--native',default='data/S2-2017-T31TFM-PixelSet')
    p.add_argument('--output',default='results/grouped_scratch')
    p.add_argument('--fold',type=int,default=1)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--parallel',type=int,default=3)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    globals()[args.command](args)

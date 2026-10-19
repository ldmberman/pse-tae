"""Sixteen fixed patch groups, full-width TAE, and frozen-baseline corrections."""
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
from torch.utils.data import Dataset, DataLoader
from models.tgt_correction import GroupedCorrection
from scripts.frozen_trajectory_probes import (NAMES, MISSING_PATCH_PIDS, partitions,
                                             standardization, metrics, focal_loss, write_json)
from scripts.train_pixel_temporal import EpochSampler, runtime

VARIANTS = ['control', 'temporal', 'spatial_nopos']


def group_patch(a):
    """4x4 cells, each an 8x8 set; all dates share exactly the same grid mapping."""
    assert a.shape == (24, 10, 32, 32) and a.dtype == np.uint16
    raw = a.reshape(24, 10, 4, 8, 4, 8).transpose(2, 4, 3, 5, 0, 1).reshape(16, 64, 24, 10)
    raw = np.ascontiguousarray(raw)
    return raw, np.any(raw != 0, axis=(-1, -2))


class GroupInputs(Dataset):
    def __init__(self, args, variant, norm=None):
        self.args, self.variant = args, variant
        with np.load(Path(args.cache) / f'fold_{args.fold}_features.npz') as f:
            self.pids, self.y, self.days = f['pids'], f['labels'], f['days']
            e, self.logits = f['embeddings'], f['logits']
        self.splits = partitions(len(self.y), args.fold)
        self.norm = standardization(e[self.splits['train']]) if norm is None else norm
        self.e = (e - self.norm[0]) / self.norm[1]
        with open(Path(args.native) / 'S2-2017-T31TFM-meanstd.pkl', 'rb') as f:
            self.mean, self.std = [np.asarray(v, np.float32) for v in pickle.load(f)]
        assert self.mean.shape == self.std.shape == (24, 10) and (self.std > 0).all()
        assert self.e.shape == (len(self.y), 128) and self.logits.shape == (len(self.y), 20)
        assert np.isfinite(self.e).all() and np.isfinite(self.logits).all()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, key):
        i = key[0] if isinstance(key, tuple) else key
        result = dict(e=self.e[i], baseline_logits=self.logits[i])
        if self.variant != 'control':
            path = Path(self.args.patch) / 'DATA' / f'{self.pids[i]}.npy'
            if not path.exists() and self.pids[i] in MISSING_PATCH_PIDS:
                raw, mask = np.zeros((16, 64, 24, 10), np.uint16), np.zeros((16, 64), bool)
            else:
                raw, mask = group_patch(np.load(path))
            # Convert uint16 on CPU; boolean GPU indexing is not supported for uint16.
            result.update(raw=raw.astype(np.float32), mask=mask)
        return result, int(self.y[i]), int(i)


def batches(ds, ids, args):
    return DataLoader(ds, sampler=ids, batch_size=args.batch_size,
                      num_workers=args.workers, pin_memory=args.device == 'cuda',
                      persistent_workers=args.workers > 0,
                      generator=torch.Generator().manual_seed(args.seed))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out, labels, indices = [], [], []
    for batch, y, ids in loader:
        logits = model(**to_device(batch, device))
        assert torch.isfinite(logits).all(), 'Nonfinite predictions'
        out.append(logits.cpu().numpy())
        labels.append(y.numpy())
        indices.append(ids.numpy())
    return np.concatenate(out), np.concatenate(labels), np.concatenate(indices)


def train(args):
    runtime(args.seed)
    folder = Path(args.output) / f'Fold_{args.fold}' / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    ds = GroupInputs(args, args.variant)
    protocol = dict(vars(args), trainable_parameters=None,
        architecture='Frozen baseline logits + MLP(128 frozen embedding + 256 group summary)',
        grouping='Fixed 4x4 patch cells, all 64 grid pixels per cell, fixed membership over 24 dates',
        temporal='Spectral 10-32-64, group mean/sample-std 128, original TAE width128 heads4 key32 MLP512-128-128',
        spatial='Optional 128-wide 4-head pre-norm self-attention, FF256, dropout .2; no coordinates or cell IDs',
        classifier='384-128-64-20, final layer zero initialized; no frozen-embedding projection',
        loss='Repository focal gamma1 detached probability, unweighted',
        optimizer=f'AdamW lr{args.lr} weight_decay.0001; plateau scheduler factor.5 patience4; gradient clip5',
        sampling='All patch pixels; no random subsampling or jitter; all-zero trajectories masked',
        pooling='Sample std N-1; singleton denominator1; empty groups excluded; groups equally weighted',
        limitations='Resized grid support is not an independent parcel mask; blended boundaries and repeated resampled pixels retained. Previously inspected fold1 is exploratory.',
        baseline=json.loads((Path(args.cache) / f'fold_{args.fold}_extraction.json').read_text()),
        pid_sha256=hashlib.sha256(','.join(ds.pids).encode()).hexdigest(),
        split_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in ds.splits.items()})
    model = GroupedCorrection(ds.days, ds.mean, ds.std, args.variant).to(args.device)
    protocol['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if (folder / 'protocol.json').exists():
        assert json.loads((folder / 'protocol.json').read_text()) == protocol, 'Existing run protocol differs'
    write_json(folder / 'protocol.json', protocol)
    if (folder / 'done.json').exists():
        print('Already complete', args.variant, flush=True)
        return
    tr, va = ds.splits['train'], ds.splits['val']
    if args.smoke:
        tr, va = tr[:128], va[:64]
    sampler = EpochSampler(tr, args.seed)
    train_loader, val_loader = batches(ds, sampler, args), batches(ds, va.tolist(), args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=4)
    started, best_epoch, start_epoch, history = time.time(), 0, 1, []

    def save_best(epoch, score, logits):
        torch.save(dict(state_dict=model.state_dict(), normalization=ds.norm, epoch=epoch,
                        validation=score, protocol=protocol), folder / 'best.tmp.pt')
        os.replace(folder / 'best.tmp.pt', folder / 'best.pt')
        np.savez_compressed(folder / 'validation_predictions.npz', indices=va, labels=ds.y[va], logits=logits)

    if (folder / 'last.pt').exists():
        ck = torch.load(folder / 'last.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        history, best, best_epoch, start_epoch = ck['history'], ck['best'], ck['best_epoch'], ck['epoch'] + 1
        torch.set_rng_state(ck['torch_rng'].cpu())
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    else:
        logits, labels, ix = predict(model, val_loader, args.device)
        assert np.array_equal(ix, va) and np.array_equal(logits, ds.logits[va]), 'Epoch zero changed baseline'
        score = metrics(labels, logits.argmax(1))
        best = score['miou']
        save_best(0, score, logits)
        history.append(dict(epoch=0, validation=score, baseline_identical=True))
        write_json(folder / 'history.json', history)
    print(json.dumps(dict(variant=args.variant, parameters=protocol['trainable_parameters'],
                          baseline_verified=True, start_epoch=start_epoch, best_validation_miou=best)), flush=True)
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch - 1 - best_epoch >= args.patience:
            break
        tick, losses, count = time.time(), 0., 0
        sampler.epoch = epoch
        model.train()
        for step, (batch, y, _) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss = focal_loss(model(**to_device(batch, args.device)), y.to(args.device))
            assert torch.isfinite(loss), 'Nonfinite training loss'
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses += loss.item() * len(y)
            count += len(y)
            if step % 400 == 0:
                print(json.dumps(dict(epoch=epoch, step=step, loss=loss.item(), gradient_norm=float(norm), seconds=time.time()-tick)), flush=True)
        logits, labels, ix = predict(model, val_loader, args.device)
        assert np.array_equal(ix, va)
        score = metrics(labels, logits.argmax(1))
        scheduler.step(score['miou'])
        history.append(dict(epoch=epoch, train_loss=losses/count, validation=score,
                            seconds=time.time()-tick, lr=optimizer.param_groups[0]['lr']))
        if score['miou'] > best:
            best, best_epoch = score['miou'], epoch
            save_best(epoch, score, logits)
        ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                  epoch=epoch, best=best, best_epoch=best_epoch, history=history,
                  torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else [])
        torch.save(ck, folder / 'last.tmp.pt')
        os.replace(folder / 'last.tmp.pt', folder / 'last.pt')
        write_json(folder / 'history.json', history)
        write_json(folder / 'status.json', dict(epoch=epoch, best_epoch=best_epoch, best_miou=best, updated_unix=time.time()))
        print(json.dumps(dict(epoch=epoch, validation_miou=score['miou'], best_epoch=best_epoch,
                              best_miou=best, seconds=time.time()-tick)), flush=True)
        completed_epoch = epoch
    write_json(folder / 'done.json', dict(epochs=completed_epoch, best_epoch=best_epoch,
        best_validation_miou=best, elapsed_seconds=time.time()-started, baseline_initialization_verified=True))


def evaluate(args):
    runtime(args.seed)
    root = Path(args.output) / f'Fold_{args.fold}'
    assert all((root / v / 'done.json').exists() for v in VARIANTS)
    ck = {v: torch.load(root / v / 'best.pt', map_location='cpu', weights_only=False) for v in VARIANTS}
    selected = max(VARIANTS, key=lambda v: ck[v]['validation']['miou'])
    selection = dict(selected=selected, criterion='validation mIoU including epoch0',
                     best_epochs={v: c['epoch'] for v, c in ck.items()})
    write_json(root / 'selection.json', selection)
    ds = GroupInputs(args, 'control')
    test = ds.splits['test'][:64] if args.smoke else ds.splits['test']
    y = ds.y[test]
    pred = {'baseline': ds.logits[test].argmax(1)}
    scores = {'baseline': metrics(y, pred['baseline'])}
    for v in VARIANTS:
        inputs = GroupInputs(args, v, ck[v]['normalization'])
        model = GroupedCorrection(inputs.days, inputs.mean, inputs.std, v).to(args.device)
        model.load_state_dict(ck[v]['state_dict'])
        logits, labels, ix = predict(model, batches(inputs, test.tolist(), args), args.device)
        assert np.array_equal(ix, test) and np.array_equal(labels, y)
        pred[v] = logits.argmax(1)
        scores[v] = metrics(y, pred[v])
        np.savez_compressed(root / v / 'test_predictions.npz', indices=ix, labels=y, logits=logits)
    comparisons = [(v, 'baseline') for v in VARIANTS] + [(v, 'control') for v in VARIANTS[1:]] + [('spatial_nopos', 'temporal')]
    delta = {f'{a}-minus-{b}': [] for a, b in comparisons}
    rng = np.random.RandomState(20261016)
    for _ in range(1000):
        ix = rng.randint(len(test), size=len(test))
        m = {v: metrics(y[ix], p[ix])['miou'] for v, p in pred.items()}
        for a, b in comparisons:
            delta[f'{a}-minus-{b}'].append(m[a] - m[b])
    report = dict(selection=selection, metrics=scores, paired_miou_differences={
        f'{a}-minus-{b}': dict(delta=scores[a]['miou']-scores[b]['miou'],
                               ci95=np.quantile(delta[f'{a}-minus-{b}'], [.025, .975]).tolist()) for a,b in comparisons},
        caveat='Exploratory fold1; parcel bootstrap conditions on these fitted models. Confirm improvements on other folds.')
    write_json(root / 'report.json', report)
    rows = ''.join(f'<tr><td>{v}</td><td>{100*s["oa"]:.3f}</td><td>{100*s["mean_recall"]:.3f}</td><td>{100*s["miou"]:.3f}</td></tr>' for v,s in scores.items())
    perclass = ''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{100*s["iou"][i]:.2f}</td>' for s in scores.values())+'</tr>' for i,name in enumerate(NAMES))
    (root / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Grouped corrections</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto}td,th{padding:8px;text-align:left}tr:nth-child(even){background:#eee}</style>'
        f'<h1>Sixteen-group full-width corrections</h1><p>Validation-selected: {selected}. {report["caveat"]}</p>'
        '<table><tr><th>Model</th><th>Accuracy %</th><th>Mean recall %</th><th>mIoU %</th></tr>'+rows+'</table>'
        '<h2>Per-class IoU %</h2><table><tr><th>Class</th>'+''.join(f'<th>{v}</th>' for v in scores)+'</tr>'+perclass+'</table>')
    print(json.dumps({v: {k:s[k] for k in ['oa', 'miou', 'mean_recall']} for v,s in scores.items()}), flush=True)


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'runner.pid').write_text(str(os.getpid()))
    common = ['--output', args.output, '--cache', args.cache, '--patch', args.patch, '--native', args.native,
              '--fold', str(args.fold), '--seed', str(args.seed), '--epochs', str(args.epochs),
              '--patience', str(args.patience), '--batch-size', str(args.batch_size), '--workers', str(args.workers),
              '--lr', str(args.lr), '--device', args.device] + (['--smoke'] if args.smoke else [])
    command = [sys.executable, '-u', str(Path(__file__).resolve())]
    complete = []
    def fit(v):
        with (root / (v+'.log')).open('a') as log:
            subprocess.run(command+['train', '--variant', v]+common, stdout=log, stderr=subprocess.STDOUT, check=True)
        return v
    try:
        write_json(root / 'status.json', dict(stage='training', complete=complete, started_unix=time.time()))
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for f in as_completed([pool.submit(fit, v) for v in VARIANTS]):
                complete.append(f.result())
                write_json(root / 'status.json', dict(stage='training', complete=complete, updated_unix=time.time()))
        write_json(root / 'status.json', dict(stage='evaluating', complete=complete, updated_unix=time.time()))
        with (root / 'evaluate.log').open('a') as log:
            subprocess.run(command+['evaluate']+common, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(root / 'status.json', dict(stage='complete', complete=complete, updated_unix=time.time(), report=f'Fold_{args.fold}/report.html'))
    except Exception as exc:
        write_json(root / 'status.json', dict(stage='failed', complete=complete, error=repr(exc), updated_unix=time.time()))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train', 'evaluate', 'run'])
    p.add_argument('--variant', choices=VARIANTS, default='temporal')
    p.add_argument('--cache', default='results/pair_probes')
    p.add_argument('--patch', default='data/s2-2017-IGARSS-NNI-NPY')
    p.add_argument('--native', default='data/S2-2017-T31TFM-PixelSet')
    p.add_argument('--output', default='results/grouped_corrections')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    globals()[args.command](args)

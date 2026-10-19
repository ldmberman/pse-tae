"""From-scratch fold comparison; no frozen embeddings or logits enter training.

Train uses fresh deterministic native-pixel samples each epoch. Validation/test
use the same fixed native samples as the cached baseline for paired comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from models.pixel_temporal import PixelTemporalClassifier
from scripts.frozen_trajectory_probes import partitions, metrics, focal_loss, write_json, NAMES

VARIANTS = ['temporal', 'temporal_spatial_nopos']


def runtime(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


class Parcels(Dataset):
    def __init__(self, args, training=False):
        self.args, self.training = args, training
        # Only metadata arrays are accessed: embeddings/logits are never loaded.
        with np.load(Path(args.features) / f'fold_{args.fold}_features.npz') as f:
            self.pids, self.y, self.days = f['pids'], f['labels'], f['days']
        self.splits = partitions(len(self.y), args.fold)
        with open(Path(args.native) / 'S2-2017-T31TFM-meanstd.pkl', 'rb') as f:
            self.mean, self.std = [np.asarray(a, np.float32) for a in pickle.load(f)]
        assert self.mean.shape == self.std.shape == (24, 10)
        assert (self.std > 0).all()
        geom = json.loads((Path(args.native) / 'META/geomfeat.json').read_text())
        g = np.array([geom[p] for p in self.pids], np.float32)
        tr = self.splits['train']
        self.geom_mean, self.geom_std = g[tr].mean(0), g[tr].std(0).clip(1e-6)
        self.geometry = (g - self.geom_mean) / self.geom_std
        assert np.isfinite(self.geometry).all()
        if not training:
            cache = Path(args.pixels)
            assert np.array_equal(np.load(cache / 'pids.npy'), self.pids)
            self.pixels = np.load(cache / 'pixels.npy', mmap_mode='r')
            self.mask = np.load(cache / 'mask.npy', mmap_mode='r')
            assert self.pixels.shape == (len(self.y), 64, 24, 10)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, key):
        if self.training:
            i, epoch = key
            rng = np.random.RandomState(np.random.SeedSequence(
                [self.args.seed, int(epoch), int(self.pids[i])]).generate_state(1)[0])
            raw = np.load(Path(self.args.native) / 'DATA' / f'{self.pids[i]}.npy')
            n = min(raw.shape[-1], 64)
            selected = rng.choice(raw.shape[-1], n, replace=False) if raw.shape[-1] > 64 else np.arange(n)
            x = np.zeros((64, 24, 10), np.float32)
            x[:n] = raw[:, :, selected].transpose(2, 0, 1)
            mask = np.arange(64) < n
            x = (x - self.mean) / self.std
            x += rng.normal(0, .01, x.shape).clip(-.05, .05).astype(np.float32)
        else:
            i = key
            x = (self.pixels[i].astype(np.float32) - self.mean) / self.std
            mask = self.mask[i].copy()
        return x, mask, self.geometry[i], self.y[i], i


class EpochSampler(Sampler):
    def __init__(self, indices, seed):
        self.indices, self.seed, self.epoch = np.asarray(indices), seed, 0

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        order = np.random.RandomState(self.seed + self.epoch).permutation(self.indices)
        return iter((int(i), self.epoch) for i in order)


def loader(ds, sampler, args):
    return DataLoader(ds, sampler=sampler, batch_size=args.batch_size,
                      num_workers=args.workers, persistent_workers=args.workers > 0,
                      pin_memory=True, generator=torch.Generator().manual_seed(args.seed))


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    ys, logits, indices = [], [], []
    for x, mask, geom, y, ix in batches:
        out = model(x.to(device), mask.to(device), geom.to(device))
        assert torch.isfinite(out).all(), 'Nonfinite evaluation logits'
        ys.append(y.numpy())
        logits.append(out.cpu().numpy())
        indices.append(ix.numpy())
    return np.concatenate(ys), np.concatenate(logits), np.concatenate(indices)


def train(args):
    runtime(args.seed)
    out = Path(args.out) / f'Fold_{args.fold}' / args.variant
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'done.json').exists():
        print('Already complete', out, flush=True)
        return
    ds, ev = Parcels(args, True), Parcels(args)
    tr, va = ds.splits['train'], ds.splits['val']
    if args.smoke:
        tr, va = tr[:256], va[:256]
    sampler = EpochSampler(tr, args.seed)
    batches, val_batches = loader(ds, sampler, args), loader(ev, va.tolist(), args)
    model = PixelTemporalClassifier(ds.days, args.variant == VARIANTS[1]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = dict(vars(args), parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  split_sizes={k: len(v) for k, v in ds.splits.items()},
                  split_hashes={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in ds.splits.items()},
                  geometry_mean=ds.geom_mean.tolist(), geometry_std=ds.geom_std.tolist(),
                  days=ds.days.tolist(), spectral_mean=ds.mean.tolist(), spectral_std=ds.std.tolist(),
                  loss='repository focal gamma=1, detached probability, no class weights',
                  training_pixels='64 native pixels, fresh per epoch, identities shared across dates; jitter .01 clipped .05',
                  evaluation_pixels='fixed native cache, baseline seed20261013, no jitter',
                  frozen_inputs=False, temporal_encoder='original TAE, per pixel; width64 heads4 key16',
                  spatial_encoder='one width64 heads4 FF128 pre-norm block, no coordinates' if args.variant == VARIANTS[1] else None)
    write_json(out / 'config.json', config)
    start_epoch, best, history = 1, -1., []
    if (out / 'last.pt').exists():
        ck = torch.load(out / 'last.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch, best, history = ck['epoch'] + 1, ck['best'], ck['history']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    print(json.dumps({'variant': args.variant, 'parameters': config['parameters'],
                      'epochs': args.epochs, 'start_epoch': start_epoch}), flush=True)
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        tick = time.time()
        sampler.epoch = epoch
        model.train()
        total, count = 0., 0
        for step, (x, mask, geom, y, _) in enumerate(batches):
            x, mask, geom, y = [a.to(args.device, non_blocking=True) for a in (x, mask, geom, y)]
            optimizer.zero_grad(set_to_none=True)
            loss = focal_loss(model(x, mask, geom), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch} step {step}')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
            if step % 200 == 0:
                print(json.dumps({'epoch': epoch, 'step': step, 'loss': loss.item(),
                                  'seconds': time.time() - tick}), flush=True)
        y, logits, _ = predict(model, val_batches, args.device)
        score = metrics(y, logits.argmax(1))
        row = dict(epoch=epoch, train_loss=total/count, validation=score, seconds=time.time()-tick)
        history.append(row)
        if score['miou'] > best:
            best = score['miou']
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'validation': score,
                        'config': config}, out / 'best.pt')
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                     best=best, history=history, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        torch.save(state, out / 'last.tmp.pt')
        os.replace(out / 'last.tmp.pt', out / 'last.pt')
        write_json(out / 'history.json', history)
        write_json(out / 'status.json', dict(stage='training', epoch=epoch, best_validation_miou=best,
                                           elapsed_seconds=time.time()-started, updated_unix=time.time()))
        print(json.dumps(row), flush=True)
    write_json(out / 'done.json', dict(stage='complete', epochs=args.epochs, best_validation_miou=best,
                                     elapsed_seconds=time.time()-started))


def evaluate(args):
    runtime(args.seed)
    root = Path(args.out) / f'Fold_{args.fold}'
    assert all((root / v / 'done.json').exists() for v in VARIANTS), 'Finish both fits before test evaluation'
    ds = Parcels(args)
    test = ds.splits['test']
    if args.smoke:
        test = test[:256]
    # Lock model selection using validation before accessing test logits/labels.
    checkpoints = {v: torch.load(root / v / 'best.pt', map_location='cpu', weights_only=False) for v in VARIANTS}
    selected = max(VARIANTS, key=lambda v: checkpoints[v]['validation']['miou'])
    write_json(root / 'selection.json', dict(selected=selected, criterion='validation mIoU',
                                            best_epochs={v: c['epoch'] for v, c in checkpoints.items()}))
    with np.load(Path(args.features) / f'fold_{args.fold}_features.npz') as f:
        baseline = f['logits'][test].argmax(1)
    scores = {'baseline': metrics(ds.y[test], baseline)}
    predictions = {'baseline': baseline}
    for v, ck in checkpoints.items():
        model = PixelTemporalClassifier(ds.days, v == VARIANTS[1]).to(args.device)
        model.load_state_dict(ck['model'])
        y, logits, ix = predict(model, loader(ds, test.tolist(), args), args.device)
        assert np.array_equal(ix, test)
        scores[v], predictions[v] = metrics(y, logits.argmax(1)), logits.argmax(1)
        np.savez_compressed(root / v / 'test_predictions.npz', indices=ix, labels=y, logits=logits)
    rng = np.random.RandomState(20261016)
    deltas = {v: [] for v in VARIANTS}
    for _ in range(1000):
        ix = rng.randint(len(test), size=len(test))
        b = metrics(ds.y[test][ix], baseline[ix])['miou']
        for v in VARIANTS:
            deltas[v].append(metrics(ds.y[test][ix], predictions[v][ix])['miou'] - b)
    report = dict(selected_by_validation=selected, metrics=scores,
                  paired_miou_delta_ci95={v: np.quantile(deltas[v], [.025, .975]).tolist() for v in VARIANTS},
                  caveat='Exploratory fold 1 already inspected in prior experiments; confirm on other folds.')
    write_json(root / 'report.json', report)
    rows = ''.join(f'<tr><td>{v}</td><td>{100*s["oa"]:.3f}</td><td>{100*s["mean_recall"]:.3f}</td><td>{100*s["miou"]:.3f}</td></tr>' for v, s in scores.items())
    classes = ''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{100*s["iou"][i]:.2f}</td>' for s in scores.values())+'</tr>' for i, name in enumerate(NAMES))
    (root / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Pixel temporal fold 1</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto}td,th{padding:8px;text-align:left}tr:nth-child(even){background:#eee}</style>'
        f'<h1>From-scratch pixel temporal attention</h1><p>Validation-selected: {selected}. {report["caveat"]}</p>'
        '<table><tr><th>Model</th><th>Accuracy %</th><th>Mean recall %</th><th>mIoU %</th></tr>'+rows+'</table>'
        '<h2>Per-class IoU %</h2><table><tr><th>Class</th>'+''.join(f'<th>{v}</th>' for v in scores)+'</tr>'+classes+'</table>')
    print(json.dumps(report), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['train', 'evaluate'])
    p.add_argument('--variant', choices=VARIANTS, default=VARIANTS[0])
    p.add_argument('--native', default='data/S2-2017-T31TFM-PixelSet')
    p.add_argument('--features', default='results/pair_probes')
    p.add_argument('--pixels', default='results/frozen_trajectories/native_cache')
    p.add_argument('--out', default='results/pixel_temporal_scratch')
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
    (train if args.command == 'train' else evaluate)(args)

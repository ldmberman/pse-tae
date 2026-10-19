"""Fold-controlled probes of pixel trajectories alongside cached PSE+TAE embeddings.

The baseline model is never loaded or updated. Native-pixel branches share the
baseline's exact pixel sample; patch branches share their own fixed sample.
Select epochs using validation mIoU only;
the separate `evaluate` command opens the test partition after all fits finish.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score

NAMES = ['Meadow', 'Winter Durum Wheat', 'Spring Cereal', 'Summer Cereal',
         'Winter Cereal', 'Sorgho/millet/moha', 'Cereal', 'Leguminous Fodder',
         'Other Fodder', 'Winter Rapeseed', 'Sunflower', 'Soy', 'Protein Crops',
         'Potato', 'Fruits/Legumes/Flowers', 'Ligneous', 'Orchard', 'Grapevine',
         'Non-Agricultural', 'Woodlands']
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
INDEX_NAMES = ['NDVI', 'NDRE5', 'NDRE6', 'NDRE7', 'NDMI', 'ND_SWIR']
INDEX_PAIRS = [(6, 2), (7, 3), (7, 4), (7, 5), (7, 8), (8, 9)]
VARIANTS = ['control', 'static', 'temporal', 'patch_temporal', 'spatial_nopos', 'spatial']
PIXEL_SEED = 20261016
CACHE_VERSION = 2
MISSING_PATCH_PIDS = {'199461', '199462', '199463', '199464'}


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def partitions(n, fold):
    order = np.arange(n)
    np.random.RandomState(1).shuffle(order)
    splits = list(KFold(5, shuffle=False).split(order))
    trainval, test = splits[fold - 1]
    nv = len(splits[0][1])
    return dict(train=order[trainval[:-nv]], val=order[trainval[-nv:]], test=order[test])


def metrics(y, pred):
    cm = np.bincount(y * 20 + pred, minlength=400).reshape(20, 20)
    tp = cm.diagonal()
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    return dict(oa=float(tp.sum() / cm.sum()), miou=float(iou.mean()),
                mean_recall=float(recall.mean()), iou=iou.tolist(),
                recall=recall.tolist(), precision=(tp / np.maximum(cm.sum(0), 1)).tolist(),
                confusion=cm.tolist())


def sample_patch(a, pid, pixels=64):
    """Group identical full trajectories; centroid coordinates remove NN duplication.

    Distinct physical pixels with identical complete spectra cannot be separated
    without source pixel IDs. No such IDs are present in the patch archive.
    """
    assert a.shape == (24, 10, 32, 32) and a.dtype == np.uint16
    rows = np.ascontiguousarray(a.reshape(240, -1).T)
    # Nonzero archive support, NOT an independently supplied parcel mask.
    # Some resized parcels contain blended boundary values: record this limitation.
    valid = np.any(rows != 0, axis=1)
    flat = np.flatnonzero(valid)
    x = np.zeros((pixels, 24, 10), np.uint16)
    xy = np.zeros((pixels, 2), np.float32)
    mask = np.zeros(pixels, bool)
    if not len(flat):
        return x, xy, mask, 0
    keys = rows[valid].view(np.dtype((np.void, rows.dtype.itemsize * 240))).ravel()
    _, first, inverse, counts = np.unique(keys, return_index=True,
                                           return_inverse=True, return_counts=True)
    coords = np.stack([(flat % 32) / 31 * 2 - 1, (flat // 32) / 31 * 2 - 1], axis=1)
    centroids = np.stack([np.bincount(inverse, weights=coords[:, k]) / counts
                          for k in range(2)], axis=1)
    rng = np.random.RandomState((PIXEL_SEED + int(pid)) % 2**32)
    select = np.sort(rng.choice(len(first), min(pixels, len(first)), replace=False))
    n = len(select)
    x[:n] = rows[valid][first[select]].reshape(n, 24, 10)
    xy[:n] = centroids[select]
    mask[:n] = True
    return x, xy, mask, len(first)


def sample_native(a, pid, pixels=64):
    assert a.shape[:2] == (24, 10) and a.dtype == np.uint16
    x = np.zeros((pixels, 24, 10), np.uint16)
    mask = np.zeros(pixels, bool)
    n = a.shape[-1]
    if n > pixels:
        # Exactly the sample used by frozen_pair_probes.FixedPixels.
        indices = np.random.RandomState((20261013 + int(pid)) % 2**32).choice(
            list(range(n)), size=pixels, replace=False)
    else:
        indices = np.arange(n)
    x[:len(indices)] = a[:, :, indices].transpose(2, 0, 1)
    mask[:len(indices)] = True
    return x, np.zeros((pixels, 2), np.float32), mask, n


def spectral_numpy(raw):
    x = raw.astype(np.float32) / 10000.
    indices = [np.divide(x[..., a] - x[..., b], x[..., a] + x[..., b],
                         out=np.zeros(x.shape[:-1], np.float32),
                         where=(x[..., a] + x[..., b]) > 1e-6) for a, b in INDEX_PAIRS]
    return np.concatenate([x, np.stack(indices, axis=-1)], axis=-1)


def static_features(raw, mask, days, dates):
    x = spectral_numpy(raw[mask])  # pixels, dates, channels
    if not len(x):
        return np.zeros(880, np.float32)
    slopes = np.diff(x, axis=1) / np.diff(days)[None, :, None]
    trapezoid = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    desc = [trapezoid(x, days, axis=1) / (days[-1] - days[0]),
            x.min(1), x.max(1), np.ptp(x, axis=1), days[x.argmax(1)] / 365.,
            slopes.min(1), slopes.max(1)]
    months = np.array([int(str(d)[4:6]) for d in dates])
    for lo, hi in [(1, 3), (4, 6), (7, 8), (9, 10)]:
        select = (months >= lo) & (months <= hi)
        assert select.any()
        desc.append(x[:, select].mean(1))
    d = np.stack(desc, axis=1)  # pixels, descriptor, channel
    return np.stack([d.mean(0), d.std(0), *np.quantile(d, [.1, .5, .9], axis=0)]).ravel().astype(np.float32)


def prepare_one(task):
    i, pid, folder, pixels, days, dates, source = task
    path = Path(folder) / 'DATA' / f'{pid}.npy'
    if not path.exists() and source == 'patch' and pid in MISSING_PATCH_PIDS:
        # Preserve every original train/validation/test parcel. Its frozen
        # embedding remains available, but the missing patch contributes no pixels.
        return (i, np.zeros((pixels,24,10), np.uint16), np.zeros((pixels,2), np.float32),
                np.zeros(pixels, bool), 0, np.zeros(880, np.float32))
    a = np.load(path)
    x, xy, mask, count = (sample_native if source == 'native' else sample_patch)(a, pid, pixels)
    return i, x, xy, mask, count, static_features(x, mask, days, dates)


def prepare(args):
    source_folder = args.native if args.source == 'native' else args.patch
    out = Path(args.trajectories if args.source == 'native' else args.patch_trajectories)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(Path(args.cache) / f'fold_{args.fold}_features.npz')
    extraction = json.loads((Path(args.cache) / f'fold_{args.fold}_extraction.json').read_text())
    if args.source == 'native':
        assert extraction['pixel_seed'] == 20261013 and extraction['npixel'] == args.pixels, \
            'Native branch must match the frozen baseline pixel sampling'
    pids, days, dates = data['pids'], data['days'], data['dates']
    if args.limit:
        pids = pids[:args.limit]
    metadata = dict(version=CACHE_VERSION, n=len(pids), pixels=args.pixels,
                    source=args.source, pixel_seed=20261013 if args.source == 'native' else PIXEL_SEED,
                    days=days.tolist(), dates=dates.tolist(),
                    pid_sha256=hashlib.sha256(','.join(pids).encode()).hexdigest(),
                    source_folder=str(Path(source_folder).resolve()))
    manifest = out / 'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        assert all(old[k] == v for k, v in metadata.items()), 'Cache protocol mismatch'
        print('CACHE COMPLETE', out, flush=True)
        return
    missing = [p for p in pids if not (Path(source_folder) / 'DATA' / f'{p}.npy').exists()]
    assert not missing or (args.source == 'patch' and set(missing) <= MISSING_PATCH_PIDS), \
        f'{len(missing)} unexpected arrays missing, including {missing[:5]}'
    metadata['missing_pids'] = list(missing)
    # Validate labels when archive metadata is available; the smoke test may run
    # while the archive is still extracting, so it explicitly omits this check.
    labels_path = Path(source_folder) / 'META' / 'labels.json'
    if not args.limit:
        labels = json.loads(labels_path.read_text())['label_44class']
        classes = [1, 3, 4, 5, 6, 8, 9, 12, 13, 14, 16, 18, 19, 23, 28, 31, 33, 34, 36, 39]
        assert np.array_equal([classes.index(labels[p]) for p in pids], data['labels'])
    maps = {}
    for key, dtype, shape in [('pixels', 'uint16', (len(pids), args.pixels, 24, 10)),
                              ('coords', 'float32', (len(pids), args.pixels, 2)),
                              ('mask', 'bool', (len(pids), args.pixels)),
                              ('distinct_counts', 'int32', (len(pids),)),
                              ('static', 'float32', (len(pids), 880))]:
        maps[key] = np.lib.format.open_memmap(out / f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
    start = time.time()
    tasks = ((i, p, source_folder, args.pixels, days, dates, args.source) for i, p in enumerate(pids))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        # executor.map materializes futures; 191k tiny task tuples remain manageable.
        for j, (i, x, xy, mask, count, static) in enumerate(pool.map(prepare_one, tasks, chunksize=64)):
            maps['pixels'][i], maps['coords'][i], maps['mask'][i] = x, xy, mask
            maps['distinct_counts'][i], maps['static'][i] = count, static
            if j % 5000 == 0:
                print('PREPARE', j, '/', len(pids), 'seconds', round(time.time()-start), flush=True)
    for a in maps.values():
        a.flush()
    np.save(out / 'pids.npy', pids)
    desc_names = ['time_weighted_mean', 'min', 'max', 'amplitude', 'peak_day_div365',
                  'min_daily_slope', 'max_daily_slope', 'Jan_Mar_mean', 'Apr_Jun_mean',
                  'Jul_Aug_mean', 'Sep_Oct_mean']
    metadata.update(seconds=time.time()-start, empty_parcels=int((maps['distinct_counts'] == 0).sum()),
                    static_dimension=880, static_names=[f'{stat}:{desc}:{band}'
                    for stat in ['mean', 'std', 'q10', 'median', 'q90']
                    for desc in desc_names for band in BANDS + INDEX_NAMES],
                    indices={n:f'({BANDS[a]}-{BANDS[b]})/({BANDS[a]}+{BANDS[b]})'
                             for n,(a,b) in zip(INDEX_NAMES, INDEX_PAIRS)},
                    sampling=('Same up-to-64 original pixels as the baseline frozen embedding; identical sample for static and temporal branches.'
                              if args.source == 'native' else 'Up to 64 distinct complete patch trajectories; identical sample for patch_temporal and both spatial branches.'),
                    coordinates=('Unavailable; unused.' if args.source == 'native' else
                                 'Centroid of resized-grid locations sharing an identical complete trajectory; normalized to [-1,1].'),
                    background=('Original parcel membership, padding masked.' if args.source == 'native' else
                                'Only all-zero trajectories excluded. Nonzero support includes blended boundary values in some patches; no independent parcel mask is available. Exact duplicate trajectories grouped, but mixed pixels cannot be mapped to individual source pixels.'),
                    units='Raw bands /10000; indices computed per pixel and date. Static temporal features first, then distributions across pixels.')
    write_json(manifest, metadata)
    print('PREPARE COMPLETE', out, flush=True)


def standardization(x):
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    std = x.std(0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def masked_moments(x, mask):
    w = mask.to(x.dtype)[..., None]
    denom = w.sum(1).clamp_min(1)
    mean = (x * w).sum(1) / denom
    var = ((x - mean[:, None]).square() * w).sum(1) / denom
    return torch.cat([mean, (var + 1e-6).sqrt()], dim=-1)


class TrajectoryEncoder(nn.Module):
    def __init__(self, days, spatial=False, positional=False, width=32):
        super().__init__()
        self.input = nn.Linear(16, width)
        self.norm = nn.LayerNorm(width)
        self.keys = nn.Linear(width, width)
        self.query = nn.Parameter(torch.randn(4, width // 4) / math.sqrt(width // 4))
        self.values = nn.Linear(width, width)
        self.output = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        days = torch.as_tensor(days, dtype=torch.float32)
        frequencies = torch.tensor([1., 2., 4., 8.]) * (2 * math.pi / 365.)
        phase = days[:, None] * frequencies[None]
        self.register_buffer('date_encoding', torch.cat([phase.sin(), phase.cos()], dim=-1))
        self.date_projection = nn.Linear(8, width, bias=False)
        self.positional = positional
        self.spatial = spatial
        if spatial:
            # Allocate the same positional module in both spatial variants;
            # removing coordinates is the only architectural ablation.
            self.position_projection = nn.Linear(8, width, bias=False)
            self.spatial_layer = nn.TransformerEncoderLayer(width, 4, 64, dropout=.1,
                                                            batch_first=True, norm_first=True,
                                                            activation='gelu')

    def forward(self, raw, coords, mask):
        x = raw.float() / 10000.
        indices = [(x[..., a] - x[..., b]) / (x[..., a] + x[..., b]).clamp_min(1e-6)
                   for a,b in INDEX_PAIRS]
        x = torch.cat([x, torch.stack(indices, dim=-1)], dim=-1)
        h = self.norm(self.input(x) + self.date_projection(self.date_encoding)[None, None])
        b, p, t, d = h.shape
        k = self.keys(h).reshape(b, p, t, 4, d // 4)
        score = (k * self.query).sum(-1) / math.sqrt(d // 4)
        weights = score.softmax(dim=2)
        values = self.values(h).reshape(b, p, t, 4, d // 4)
        z = self.output((values * weights[..., None]).sum(2).reshape(b, p, d))
        if self.spatial:
            if self.positional:
                phase = coords[..., None] * torch.tensor([math.pi, 2*math.pi], device=coords.device)
                position = torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(-2)
                z = z + self.position_projection(position)
            safe_mask = mask.clone()
            safe_mask[:, 0] |= ~safe_mask.any(1)  # prevent all-masked softmax on empty parcels
            z = self.spatial_layer(z, src_key_padding_mask=~safe_mask)
        return masked_moments(z, mask)


class Probe(nn.Module):
    def __init__(self, variant, days):
        super().__init__()
        self.variant = variant
        self.embedding = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU())
        if variant == 'static':
            self.branch = nn.Sequential(nn.Linear(880, 64), nn.LayerNorm(64), nn.GELU())
        elif variant != 'control':
            self.branch = TrajectoryEncoder(days, spatial=variant.startswith('spatial'),
                                             positional=variant == 'spatial')
        self.branch_norm = nn.LayerNorm(64)
        # Identical classifier initialization across branches despite their
        # different numbers of randomly initialized encoder parameters.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(107)
            self.classifier = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, 20))

    def forward(self, e, raw=None, coords=None, mask=None, static=None):
        a = self.embedding(e)
        if self.variant == 'control':
            b = torch.zeros_like(a)
        elif self.variant == 'static':
            b = self.branch_norm(self.branch(static))
        else:
            b = self.branch_norm(self.branch(raw, coords, mask))
        return self.classifier(torch.cat([a, b], dim=-1))


class Inputs:
    def __init__(self, args, variant, norm=None):
        self.data = np.load(Path(args.cache) / f'fold_{args.fold}_features.npz')
        self.y = self.data['labels']
        self.splits = partitions(len(self.y), args.fold)
        self.variant = variant
        tr = self.splits['train']
        self.e = self.data['embeddings']
        self.norm = {} if norm is None else norm
        if norm is None:
            self.norm['embedding'] = standardization(self.e[tr])
        self.e = (self.e - self.norm['embedding'][0]) / self.norm['embedding'][1]
        self.arrays = {}
        if variant != 'control':
            folder = Path(args.patch_trajectories if variant in ['patch_temporal', 'spatial_nopos', 'spatial'] else args.trajectories)
            manifest = json.loads((folder / 'manifest.json').read_text())
            assert manifest['version'] == CACHE_VERSION and manifest['n'] == len(self.y)
            assert np.array_equal(np.load(folder / 'pids.npy'), self.data['pids'])
            assert manifest['days'] == self.data['days'].tolist()
            if variant == 'static':
                x = np.load(folder / 'static.npy', mmap_mode='r')
                if norm is None:
                    self.norm['static'] = standardization(x[tr])
                self.arrays['static'] = (x - self.norm['static'][0]) / self.norm['static'][1]
            else:
                for name in ['pixels', 'coords', 'mask']:
                    self.arrays[name] = np.load(folder / f'{name}.npy', mmap_mode='r')

    def batch(self, ids, device):
        result = {'e': torch.from_numpy(self.e[ids]).to(device)}
        for key, x in self.arrays.items():
            a = x[ids]
            # uint16 CUDA indexing/conversion support varies; convert the tiny batch on CPU.
            if key == 'pixels':
                a = a.astype(np.float32)
            result['raw' if key == 'pixels' else key] = torch.from_numpy(a).to(device)
        return result, torch.as_tensor(self.y[ids], device=device, dtype=torch.long)


def focal_loss(logits, y):
    # Exactly the repository baseline's gamma=1, detached-probability focal loss.
    logp = F.log_softmax(logits, dim=-1).gather(1, y[:, None]).squeeze(1)
    return (-logp * (1 - logp.detach().exp())).mean()


def predict(model, inputs, ids, args):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(ids), args.batch_size):
            batch, _ = inputs.batch(ids[start:start+args.batch_size], args.device)
            predictions.append(model(**batch).cpu().numpy())
    return np.concatenate(predictions)


def fit(args):
    folder = Path(args.output) / f'Fold_{args.fold}' / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    extraction = json.loads((Path(args.cache) / f'fold_{args.fold}_extraction.json').read_text())
    protocol = dict(variant=args.variant, fold=args.fold, seed=args.seed, epochs=args.epochs,
                    patience=args.patience, batch_size=args.batch_size, lr=args.lr,
                    loss='Repository focal gamma=1, detached probability, no class weights',
                    weight_decay=1e-4, baseline=extraction,
                    selection='Best validation mIoU; test not evaluated during fitting.',
                    trajectory_manifest_sha256=None)
    if args.variant != 'control':
        cache_folder = args.patch_trajectories if args.variant in ['patch_temporal', 'spatial_nopos', 'spatial'] else args.trajectories
        protocol['trajectory_manifest_sha256'] = hashlib.sha256(
            (Path(cache_folder) / 'manifest.json').read_bytes()).hexdigest()
    if (folder / 'protocol.json').exists():
        assert json.loads((folder / 'protocol.json').read_text()) == protocol, 'Fit protocol mismatch'
    write_json(folder / 'protocol.json', protocol)
    if (folder / 'fit_complete.json').exists():
        print('FIT ALREADY COMPLETE', args.variant, flush=True)
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    inputs = Inputs(args, args.variant)
    model = Probe(args.variant, inputs.data['days']).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=4)
    tr, va = inputs.splits['train'], inputs.splits['val']
    rng = np.random.RandomState(args.seed)
    best, best_epoch, history = -1., 0, []
    start = time.time()
    print('FIT', args.variant, 'parameters', sum(p.numel() for p in model.parameters()), flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        order = rng.permutation(tr)
        losses = 0.
        for offset in range(0, len(order), args.batch_size):
            ids = order[offset:offset+args.batch_size]
            batch, y = inputs.batch(ids, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = focal_loss(model(**batch), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses += loss.item() * len(ids)
        logits = predict(model, inputs, va, args)
        val = metrics(inputs.y[va], logits.argmax(1))
        scheduler.step(val['miou'])
        record = dict(epoch=epoch, train_loss=losses/len(tr), val=val,
                      seconds=time.time()-epoch_start, lr=optimizer.param_groups[0]['lr'])
        history.append(record)
        write_json(folder / 'history.json', history)
        if val['miou'] > best:
            best, best_epoch = val['miou'], epoch
            artifact = dict(state_dict=model.state_dict(), normalization=inputs.norm,
                            protocol=protocol, epoch=epoch, val=val)
            torch.save(artifact, folder / 'best.tmp')
            (folder / 'best.tmp').replace(folder / 'best.pt')
            np.savez_compressed(folder / 'val_predictions.npz', logits=logits,
                                labels=inputs.y[va], indices=va, pids=inputs.data['pids'][va])
        print(args.variant, 'epoch', epoch, 'val', round(val['oa']*100, 3), round(val['miou']*100, 3),
              'best', best_epoch, round(best*100, 3), 'seconds', round(record['seconds'], 1), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    write_json(folder / 'fit_complete.json', dict(best_epoch=best_epoch, val_miou=best,
               epochs=len(history), seconds=time.time()-start,
               parameters=sum(p.numel() for p in model.parameters())))


def evaluate(args, model_class=Probe, input_class=Inputs):
    root = Path(args.output) / f'Fold_{args.fold}'
    # Freeze all model choices before opening test predictions.
    for name in VARIANTS:
        assert (root / name / 'fit_complete.json').exists(), f'Incomplete fit: {name}'
    selected = {name: json.loads((root / name / 'fit_complete.json').read_text()) for name in VARIANTS}
    best_variant = max(selected, key=lambda k: selected[k]['val_miou'])
    baseline_data = np.load(Path(args.cache) / f'fold_{args.fold}_features.npz')
    va = partitions(len(baseline_data['labels']), args.fold)['val']
    baseline_val = metrics(baseline_data['labels'][va], baseline_data['logits'][va].argmax(1))
    winner = best_variant if selected[best_variant]['val_miou'] > baseline_val['miou'] else 'original_decoder'
    # Diagnostic pairs are chosen from validation errors, never from test results.
    cm = np.asarray(baseline_val['confusion']).copy()
    np.fill_diagonal(cm, 0)
    pairs = [(i, int(cm[i].argmax())) for i in range(20)
             if baseline_val['iou'][i] < .5 and cm[i].sum() > 0]
    write_json(root / 'selection.json', dict(winner=winner, best_new_variant=best_variant,
               fits=selected, baseline_val=baseline_val, weak_pairs=pairs,
               weak_pair_rule='Baseline validation IoU < .5; largest validation off-diagonal confusion.'))
    results = []
    predictions = {}
    all_logits = {}
    for name in VARIANTS:
        artifact = torch.load(root / name / 'best.pt', map_location='cpu', weights_only=False)
        inputs = input_class(args, name, norm=artifact['normalization'])
        model = model_class(name, inputs.data['days']).to(args.device)
        model.load_state_dict(artifact['state_dict'])
        te = inputs.splits['test']
        logits = predict(model, inputs, te, args)
        np.savez_compressed(root / name / 'test_predictions.npz', logits=logits,
                            labels=inputs.y[te], indices=te, pids=inputs.data['pids'][te])
        predictions[name] = logits.argmax(1)
        all_logits[name] = logits
        results.append(dict(name=name, val=artifact['val'], test=metrics(inputs.y[te], predictions[name]),
                            best_epoch=artifact['epoch'], parameters=selected[name]['parameters']))
    y = inputs.y[te]
    predictions['original_decoder'] = inputs.data['logits'][te].argmax(1)
    all_logits['original_decoder'] = inputs.data['logits'][te]
    va = inputs.splits['val']
    results.insert(0, dict(name='original_decoder', test=metrics(y, predictions['original_decoder']),
                          val=metrics(inputs.y[va], inputs.data['logits'][va].argmax(1))))
    # Paired parcel bootstrap; uncertainty conditional on this one trained fold.
    rng = np.random.RandomState(20261016)
    boot = {name: {'vs_baseline': [], 'vs_control': [], 'vs_native_temporal': [],
                   'vs_patch_temporal': []} for name in VARIANTS}
    for _ in range(500):
        ix = rng.randint(0, len(y), len(y))
        base = metrics(y[ix], predictions['original_decoder'][ix])['miou']
        control = metrics(y[ix], predictions['control'][ix])['miou']
        native_temporal = metrics(y[ix], predictions['temporal'][ix])['miou']
        patch_temporal = metrics(y[ix], predictions['patch_temporal'][ix])['miou']
        for name in VARIANTS:
            value = metrics(y[ix], predictions[name][ix])['miou']
            boot[name]['vs_baseline'].append(value-base)
            boot[name]['vs_control'].append(value-control)
            boot[name]['vs_native_temporal'].append(value-native_temporal)
            boot[name]['vs_patch_temporal'].append(value-patch_temporal)
    intervals = {name:{key:np.quantile(v, [.025,.975]).tolist() for key,v in vals.items()}
                 for name, vals in boot.items()}
    pair_results = []
    for weak, confounder in pairs:
        keep = (y == weak) | (y == confounder)
        positive = (y[keep] == weak).astype(int)
        scores = {}
        for name, logits in all_logits.items():
            score = logits[keep, weak] - logits[keep, confounder]
            scores[name] = (dict(auc=float(roc_auc_score(positive, score)),
                                 ap=float(average_precision_score(positive, score)))
                            if len(np.unique(positive)) == 2 else dict(auc=None, ap=None))
        pair_results.append(dict(weak=NAMES[weak], confounder=NAMES[confounder],
                                 positive=int(positive.sum()), negative=int((1-positive).sum()), scores=scores))
    write_json(root / 'pair_diagnostics.json', dict(pairs=pair_results,
               score='Weak-class logit minus confounder logit; pair restricted to true members for diagnosis only, no deployed pair gate.'))
    write_json(root / 'results.json', dict(winner=winner, best_new_variant=best_variant, classes=NAMES, models=results,
               miou_delta_95ci=intervals, caveat='One-fold screening; bootstrap does not measure training-seed or fold variability.'))
    rows = []
    for r in results:
        rows.append('<tr><td>'+r['name']+'</td>'+''.join(f'<td>{100*r[s][k]:.2f}</td>'
                    for s,k in [('val','miou'),('test','oa'),('test','mean_recall'),('test','miou')])+'</tr>')
    per_class = []
    for i, c in enumerate(NAMES):
        per_class.append('<tr><td>'+c+'</td>'+''.join(f'<td>{100*r["test"]["iou"][i]:.2f}</td>' for r in results)+'</tr>')
    html = '<!doctype html><meta charset="utf-8"><title>Frozen trajectory probes</title>'
    html += '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto}td,th{padding:8px;text-align:right;border-bottom:1px solid #ddd}td:first-child{text-align:left}</style>'
    html += f'<h1>Frozen baseline + pixel trajectories: fold {args.fold}</h1><p>Validation-selected variant: {winner}. One-fold screening, not a confirmed five-fold gain.</p>'
    html += '<p>Static and temporal branches use the original parcel pixels, matching the frozen baseline sample. Patch-temporal and both spatial branches use the same resized-patch sample. Compare spatial with patch-temporal to assess spatial interactions, and spatial with spatial_nopos to assess coordinates. Some patches contain blended boundary values; their nonzero support is not an independent parcel mask.</p>'
    html += '<table><tr><th>Model</th><th>Validation mIoU</th><th>Test OA</th><th>Test mean recall</th><th>Test mIoU</th></tr>'+''.join(rows)+'</table>'
    html += '<h2>Paired mIoU differences</h2><p>Percentage points, with 95% paired parcel-bootstrap intervals. These intervals condition on this trained fold; they do not measure variability across training seeds or folds.</p><table><tr><th>Model</th><th>Versus original decoder</th><th>Versus new head only</th><th>Versus patch temporal</th></tr>'
    by_name = {r['name']:r for r in results}
    for name in VARIANTS:
        html += '<tr><td>'+name+'</td>'
        for key, reference in [('vs_baseline','original_decoder'),('vs_control','control'),('vs_patch_temporal','patch_temporal')]:
            delta = 100 * (by_name[name]['test']['miou'] - by_name[reference]['test']['miou'])
            lo, hi = [100*v for v in intervals[name][key]]
            html += f'<td>{delta:+.2f} [{lo:+.2f}, {hi:+.2f}]</td>'
        html += '</tr>'
    html += '</table>'
    html += '<h2>Test per-class IoU</h2><table><tr><th>Class</th>'+''.join('<th>'+r['name']+'</th>' for r in results)+'</tr>'+''.join(per_class)+'</table>'
    html += '<h2>Weak-class separation: pairwise test ROC AUC</h2><p>Pairs chosen from baseline validation errors. Each score compares the weak-class and confounder logits on true pair members; this diagnostic is not a deployed classifier.</p><table><tr><th>Weak class / confounder</th>'+''.join('<th>'+r['name']+'</th>' for r in results)+'</tr>'
    for pair in pair_results:
        html += '<tr><td>'+pair['weak']+' / '+pair['confounder']+'</td>'
        for r in results:
            auc = pair['scores'][r['name']]['auc']
            html += '<td>'+('N/A' if auc is None else f'{auc:.3f}')+'</td>'
        html += '</tr>'
    html += '</table>'
    (root / 'report.html').write_text(html)
    print('COMPLETE', root, 'winner', winner, flush=True)
    for r in results:
        print(r['name'], 'VAL', r['val']['miou'], 'TEST', r['test']['oa'], r['test']['miou'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'fit', 'evaluate'])
    p.add_argument('--cache', default='results/pair_probes')
    p.add_argument('--trajectories', default='results/frozen_trajectories/native_cache')
    p.add_argument('--patch-trajectories', default='results/frozen_trajectories/patch_cache')
    p.add_argument('--source', choices=['native', 'patch'], default='native')
    p.add_argument('--native', default='data/S2-2017-T31TFM-PixelSet')
    p.add_argument('--patch', default='data/s2-2017-IGARSS-NNI-NPY')
    p.add_argument('--output', default='results/frozen_trajectories')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--variant', choices=VARIANTS, default='control')
    p.add_argument('--pixels', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, default=0, help='Preparation smoke-test only, separate cache required')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    globals()[args.command](args)


if __name__ == '__main__':
    main()

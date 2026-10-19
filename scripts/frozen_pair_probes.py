"""Leakage-controlled pairwise probes of frozen PSE+TAE and spectral summaries.

Run from the repository root. This script never updates a baseline checkpoint.
The original train/validation/test partition is reconstructed with seed 1.
CPU-only extraction and fitting keep the GPU available for training jobs.
"""

import os

if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import argparse
import csv
import hashlib
import html
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import PixelSetData
from models.stclassifier import PseTae
from learning.splits import partitions

CLASSES = [1, 3, 4, 5, 6, 8, 9, 12, 13, 14, 16, 18, 19, 23, 28, 31, 33, 34, 36, 39]
NAMES = ['Meadow', 'Winter Durum Wheat', 'Spring Cereal', 'Summer Cereal',
         'Winter Cereal', 'Sorgho/millet/moha', 'Cereal', 'Leguminous Fodder',
         'Other Fodder', 'Winter Rapeseed', 'Sunflower', 'Soy', 'Protein Crops',
         'Potato', 'Fruits/Legumes/Flowers', 'Ligneous', 'Orchard', 'Grapevine',
         'Non-Agricultural', 'Woodlands']
PAIRS = [(1, 4), (19, 0), (16, 0), (6, 4), (8, 0), (15, 0),
         (5, 3), (7, 0), (13, 18), (12, 4), (18, 0)]
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
INDEX_BANDS = [('NDVI', 'B8', 'B4'), ('NDRE5', 'B8A', 'B5'),
               ('NDRE6', 'B8A', 'B6'), ('NDRE7', 'B8A', 'B7'),
               ('NDMI', 'B8A', 'B11'), ('ND_SWIR', 'B11', 'B12')]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


class FixedPixels(PixelSetData):
    def __getitem__(self, item):
        # A parcel gets the same pixel sample independently of batch/worker order.
        np.random.seed((20261013 + int(self.pid[item])) % (2 ** 32))
        x, y = super().__getitem__(item)
        return x, y, item


def make_dataset(args):
    norm = pickle.load(open(Path(args.dataset) / 'S2-2017-T31TFM-meanstd.pkl', 'rb'))
    return FixedPixels(args.dataset, labels='label_44class', npixel=64,
                       sub_classes=CLASSES, norm=norm, extra_feature='geomfeat', jitter=None)


def extract(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dt = make_dataset(args)
    ckpt_path = Path(args.baseline) / f'Fold_{args.fold}' / 'model.pth.tar'
    metadata = {'fold': args.fold, 'n': len(dt), 'pixel_seed': 20261013,
                'jitter': False, 'npixel': 64, 'checkpoint_sha256': hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
                'pid_sha256': hashlib.sha256(','.join(dt.pid).encode()).hexdigest()}
    cache = out / f'fold_{args.fold}_features.npz'
    if cache.exists():
        previous = json.loads((out / f'fold_{args.fold}_extraction.json').read_text())
        assert all(previous[k] == v for k, v in metadata.items()), 'Cache provenance mismatch'
        print('USING CACHE', cache, flush=True)
        return
    model = PseTae(positions=dt.date_positions)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    metadata['checkpoint_epoch'] = int(ckpt['epoch'])
    emb = np.empty((len(dt), 128), np.float32)
    logits = np.empty((len(dt), 20), np.float32)
    loader = torch.utils.data.DataLoader(dt, batch_size=128, shuffle=False,
                                        num_workers=args.workers)
    start = time.time()
    with torch.inference_mode():
        for step, (x, y, ids) in enumerate(loader):
            h = model.temporal_encoder(model.spatial_encoder(x))
            z = model.decoder(h)
            if step == 0:
                assert torch.allclose(z, model(x), atol=1e-6)
                assert np.array_equal(y.numpy(), np.asarray(dt.target)[ids.numpy()])
            emb[ids.numpy()] = h.numpy()
            logits[ids.numpy()] = z.numpy()
            if step % 100 == 0:
                print('EXTRACT', args.fold, step, '/', len(loader), 'seconds', round(time.time()-start), flush=True)
    np.savez(cache, embeddings=emb, logits=logits, labels=np.asarray(dt.target),
             pids=np.asarray(dt.pid), dates=np.asarray(dt.dates), days=np.asarray(dt.date_positions))
    metadata['seconds'] = time.time() - start
    dump(out / f'fold_{args.fold}_extraction.json', metadata)
    print('EXTRACTION DONE', metadata, flush=True)


def spectral_features(args, data):
    out = Path(args.output)
    path = out / 'spectral_features.npz'
    if path.exists():
        cached = np.load(path)
        assert np.array_equal(cached['pids'], data['pids'])
        return cached['bands'], cached['spectral']
    moments = np.load(Path(args.moments) / 'moments.npy', mmap_mode='r')
    labels = np.load(Path(args.moments) / 'targets.npy')
    assert moments.shape[:3] == (len(data['labels']), 24, 10)
    assert np.array_equal(labels, data['labels'])
    # The cache's producer sorts parcel IDs exactly as PixelSetData does.
    mu = np.array(moments[..., 0], dtype=np.float32)
    sd = np.array(moments[..., 1], dtype=np.float32)
    # Existing moment extractor zeroed parcels with <2 pixels; recover their mean.
    zeros = np.flatnonzero(np.all(mu == 0, axis=(1, 2)))
    for i in zeros:
        x = np.load(Path(args.dataset) / 'DATA' / f'{data["pids"][i]}.npy').astype(np.float32)
        if x.shape[-1]:
            mu[i], sd[i] = x.mean(-1), x.std(-1, ddof=0)
    # Check cache indexing and formulas independently on actual parcels.
    for i in np.linspace(0, len(mu)-1, 20, dtype=int):
        x = np.load(Path(args.dataset) / 'DATA' / f'{data["pids"][i]}.npy').astype(np.float64)
        if x.shape[-1]:
            assert np.allclose(mu[i], x.mean(-1), atol=0.01)
            assert np.allclose(sd[i], x.std(-1), atol=0.01)
    geom_dict = json.load(open(Path(args.dataset) / 'META' / 'geomfeat.json'))
    geom = np.array([geom_dict[str(pid)] for pid in data['pids']], np.float32)
    band = np.concatenate([mu.reshape(len(mu), -1), sd.reshape(len(mu), -1), geom], axis=1)
    names = [f'{stat}:{date}:{band}' for stat in ['mean', 'std'] for date in data['dates'] for band in BANDS]
    names += [f'geometry:{i}' for i in range(geom.shape[1])]
    indices = []
    for name, a, b in INDEX_BANDS:
        av, bv = mu[..., BANDS.index(a)], mu[..., BANDS.index(b)]
        indices.append(np.divide(av-bv, av+bv, out=np.zeros_like(av), where=(av+bv)>1e-6))
    ind = np.stack(indices, axis=2)  # parcels x dates x indices
    days = data['days'].astype(np.float64)
    slopes = np.diff(ind, axis=1) / np.diff(days)[None, :, None]
    # Time-weighted mean and amplitude/peak timing, plus four season windows.
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    seasonal = [integrate(ind, days, axis=1)/(days[-1]-days[0]), ind.min(1), ind.max(1),
                np.ptp(ind, axis=1), days[ind.argmax(1)]/365., slopes.max(1), slopes.min(1)]
    desc_names = ['time_weighted_mean', 'minimum', 'maximum', 'amplitude', 'peak_day_div365', 'max_daily_slope', 'min_daily_slope']
    dates = data['dates'].astype(str)
    months = np.array([int(d[4:6]) for d in dates])
    for lo, hi in [(1,3), (4,6), (7,8), (9,10)]:
        seasonal.append(ind[:, (months >= lo) & (months <= hi)].mean(1))
        desc_names.append(f'months_{lo}_{hi}_mean')
    extra = np.concatenate([ind.reshape(len(mu), -1), slopes.reshape(len(mu), -1), *seasonal], axis=1).astype(np.float32)
    enames = [f'{name}:{date}' for date in data['dates'] for name, _, _ in INDEX_BANDS]
    enames += [f'daily_change:{name}:{data["dates"][t]}-{data["dates"][t+1]}' for t in range(23) for name, _, _ in INDEX_BANDS]
    enames += [f'{desc}:{name}' for desc in desc_names for name, _, _ in INDEX_BANDS]
    spectral = np.concatenate([band, extra], axis=1)
    assert np.isfinite(spectral).all()
    np.savez(path, bands=band, spectral=spectral, pids=data['pids'])
    dump(out / 'feature_definitions.json', {'band_order': BANDS, 'band_order_note':
         'Standard Sentinel-2 spectral ordering excluding B1/B9/B10, as in the paper; archive META has no explicit band-name file.',
         'band_features': names, 'additional_features': enames,
         'indices': {name: f'(mean_{a}-mean_{b})/(mean_{a}+mean_{b})' for name, a, b in INDEX_BANDS},
         'spatial_summary': 'All available parcel pixels: population mean and std; original stored units. Indices are ratios of parcel means, not means of pixel ratios.',
         'temporal_summary': 'All 24 dates retained. Adjacent differences divided by actual days; trapezoidal time-weighted mean; min, max, range, peak days since first acquisition/365, extreme daily slopes and four seasonal means.',
         'scaling': 'StandardScaler fitted on pair training data only for logistic regression.',
         'zero_cache_rows_recomputed': int(len(zeros)), 'bands_dimension': int(band.shape[1]),
         'spectral_dimension': int(spectral.shape[1])})
    return band, spectral


def binary_metrics(y, score, threshold):
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return {'ap': float(average_precision_score(y, score)), 'roc_auc': float(roc_auc_score(y, score)),
            'precision': tp/max(tp+fp, 1), 'recall': tp/max(tp+fn, 1), 'iou': tp/max(tp+fp+fn, 1),
            'tp': tp, 'fp': fp, 'fn': fn, 'positive': int(y.sum()), 'negative': int((y == 0).sum()),
            'threshold': float(threshold)}


def select_threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    ious = precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1]-precision[:-1]*recall[:-1], 1e-15)
    return float(thresholds[int(np.argmax(ious))])


def multiclass_metrics(y, pred):
    cm = np.bincount(y*20+pred, minlength=400).reshape(20, 20)
    tp = cm.diagonal()
    den = cm.sum(0)+cm.sum(1)-tp
    return {'oa': float((pred == y).mean()), 'miou': float(np.mean(tp/np.maximum(den, 1)))}


def operational_threshold(y, base_pred, score, weak, conf):
    # Change only parcels whose baseline argmax is one of the two pair classes.
    gate = (base_pred == weak) | (base_pred == conf)
    base = multiclass_metrics(y, base_pred)
    best = {'enabled': False, 'threshold': None, **base}
    for th in np.unique(np.quantile(score[gate], np.linspace(0, 1, 201))):
        pred = base_pred.copy()
        pred[gate] = np.where(score[gate] >= th, weak, conf)
        met = multiclass_metrics(y, pred)
        if met['miou'] > best['miou']:
            best = {'enabled': True, 'threshold': float(th), **met}
    return best


def ap_bootstrap(y, scores, base_score, seed, reps=200):
    rng = np.random.RandomState(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = []
    for _ in range(reps):
        ix = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        deltas.append(average_precision_score(y[ix], scores[ix])-average_precision_score(y[ix], base_score[ix]))
    return [float(v) for v in np.percentile(deltas, [2.5, 97.5])]


def fit(args):
    out = Path(args.output)
    data = np.load(out / f'fold_{args.fold}_features.npz')
    split = partitions(len(data['labels']), args.fold)
    cached_val = Path(args.baseline).parent / 'logit_adjust' / f'fold_{args.fold}_logits.npz'
    if cached_val.exists():
        assert np.array_equal(split['val'], np.load(cached_val)['val_indices']), 'Split mismatch'
    assert not set(split['train']) & set(split['val'])
    assert not set(split['train']) & set(split['test'])
    band, spectral = spectral_features(args, data)
    embeddings = data['embeddings']
    labels, logits = data['labels'], data['logits']
    features = {'embedding': embeddings, 'bands': band, 'spectral': spectral,
                'embedding_bands': np.concatenate([embeddings, band], axis=1),
                'embedding_spectral': np.concatenate([embeddings, spectral], axis=1)}
    fold_out = out / f'Fold_{args.fold}'
    fold_out.mkdir(exist_ok=True)
    result = []
    for weak, conf in PAIRS:
        key = f'{weak}_{conf}'
        dest = fold_out / key
        dest.mkdir(exist_ok=True)
        if (dest / 'results.json').exists():
            result.append(json.loads((dest / 'results.json').read_text()))
            print('PAIR CACHED', key, flush=True)
            continue
        ix = {s: a[(labels[a] == weak) | (labels[a] == conf)] for s, a in split.items()}
        ys = {s: (labels[a] == weak).astype(np.int32) for s, a in ix.items()}
        counts = {s: {'weak': int(y.sum()), 'confounder': int((y == 0).sum())} for s, y in ys.items()}
        print('PAIR', NAMES[weak], 'vs', NAMES[conf], counts, flush=True)
        base_scores = {s: logits[a, weak]-logits[a, conf] for s, a in ix.items()}
        threshold = select_threshold(ys['val'], base_scores['val'])
        rec = {'weak': NAMES[weak], 'confounder': NAMES[conf], 'weak_index': weak, 'conf_index': conf,
               'counts': counts, 'baseline': {s: binary_metrics(ys[s], base_scores[s], threshold) for s in ['val','test']},
               'baseline_default_test': binary_metrics(ys['test'], base_scores['test'], 0.), 'models': []}
        scores_out = {'val_labels': ys['val'], 'test_labels': ys['test'],
                      'baseline_val': base_scores['val'], 'baseline_test': base_scores['test']}
        fitted = {}
        for fname, matrix in features.items():
            scaler = StandardScaler().fit(matrix[ix['train']])
            xt = scaler.transform(matrix[ix['train']]); xv = scaler.transform(matrix[ix['val']])
            best = None
            for weight in [None, 'balanced']:
                for c in [0.01, 0.1, 1.0]:
                    model = LogisticRegression(C=c, class_weight=weight, solver='lbfgs', max_iter=500, tol=1e-4)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always', ConvergenceWarning)
                        model.fit(xt, ys['train'])
                    s = model.decision_function(xv)
                    ap = average_precision_score(ys['val'], s)
                    candidate = {'model': model, 'scaler': scaler, 'val_ap': float(ap), 'C': c,
                                 'class_weight': weight, 'converged': not any(issubclass(w.category, ConvergenceWarning) for w in caught)}
                    if best is None or ap > best['val_ap']:
                        best = candidate
            model, scaler = best.pop('model'), best.pop('scaler')
            name = fname + '_logistic'
            sv = model.decision_function(scaler.transform(matrix[ix['val']]))
            st = model.decision_function(scaler.transform(matrix[ix['test']]))
            th = select_threshold(ys['val'], sv)
            entry = {'name': name, 'features': fname, 'family': 'logistic', 'dimension': int(matrix.shape[1]),
                     'configuration': best, 'val': binary_metrics(ys['val'], sv, th), 'test': binary_metrics(ys['test'], st, th)}
            rec['models'].append(entry)
            fitted[name] = (model, scaler, matrix)
            scores_out[name+'_val'], scores_out[name+'_test'] = sv, st
            joblib.dump({'model': model, 'scaler': scaler, 'threshold': th}, dest / f'{name}.joblib')
            print('FIT', key, name, 'val AP', round(entry['val']['ap'],4), 'test AP', round(entry['test']['ap'],4), flush=True)
        # A small nonlinear probe checks whether linear probes miss useful interactions.
        from xgboost import XGBClassifier
        for fname in ['embedding', 'spectral', 'embedding_spectral']:
            matrix = features[fname]
            best = None
            for depth in [2, 3]:
                model = XGBClassifier(n_estimators=200, max_depth=depth, learning_rate=.05,
                    min_child_weight=5, reg_lambda=10, subsample=.9, colsample_bytree=.8,
                    tree_method='hist', n_jobs=args.threads, random_state=1,
                    scale_pos_weight=float(np.sqrt((ys['train']==0).sum()/ys['train'].sum())))
                model.fit(matrix[ix['train']], ys['train'])
                sv = model.predict_proba(matrix[ix['val']])[:,1]
                ap = average_precision_score(ys['val'], sv)
                if best is None or ap > best[0]: best = (ap, model, depth)
            _, model, depth = best
            name = fname+'_xgb'
            sv = model.predict_proba(matrix[ix['val']])[:,1]
            st = model.predict_proba(matrix[ix['test']])[:,1]
            th = select_threshold(ys['val'], sv)
            entry = {'name': name, 'features': fname, 'family': 'xgb', 'dimension': int(matrix.shape[1]),
                     'configuration': {'depth': depth, 'estimators': 200},
                     'val': binary_metrics(ys['val'], sv, th), 'test': binary_metrics(ys['test'], st, th)}
            rec['models'].append(entry)
            fitted[name] = (model, None, matrix)
            scores_out[name+'_val'], scores_out[name+'_test'] = sv, st
            joblib.dump({'model': model, 'threshold': th}, dest / f'{name}.joblib')
            print('FIT', key, name, 'val AP', round(entry['val']['ap'],4), 'test AP', round(entry['test']['ap'],4), flush=True)
        selected = max(rec['models'], key=lambda r:r['val']['ap'])
        rec['selected_by_validation_ap'] = selected['name']
        rec['selected_test_ap_delta_ci95'] = ap_bootstrap(ys['test'], scores_out[selected['name']+'_test'], base_scores['test'], 100+weak)
        rec['combined_vs_embedding_logistic_test_ap_delta_ci95'] = ap_bootstrap(ys['test'], scores_out['embedding_spectral_logistic_test'], scores_out['embedding_logistic_test'], 200+weak)
        # Check effects on all 20 classes, including third-class false positives.
        model, scaler, matrix = fitted[selected['name']]
        fullscores = {}
        for s in ['val', 'test']:
            xx = matrix[split[s]]
            fullscores[s] = model.decision_function(scaler.transform(xx)) if scaler is not None else model.predict_proba(xx)[:,1]
        bp = {s: logits[split[s]].argmax(1) for s in ['val','test']}
        op = operational_threshold(labels[split['val']], bp['val'], fullscores['val'], weak, conf)
        pred = bp['test'].copy()
        gate = (pred == weak) | (pred == conf)
        if op['enabled']:
            pred[gate] = np.where(fullscores['test'][gate] >= op['threshold'], weak, conf)
        rec['operational'] = {'val_choice': op, 'test_baseline': multiclass_metrics(labels[split['test']],bp['test']),
                              'test_rerouted': multiclass_metrics(labels[split['test']], pred),
                              'test_changed': int((pred != bp['test']).sum()),
                              'test_gate_other_class_count': int(np.sum(gate & ~np.isin(labels[split['test']], [weak,conf])))}
        np.savez(dest / 'scores.npz', **scores_out)
        dump(dest / 'results.json', rec)
        result.append(rec)
        dump(fold_out / 'results.json', result)
    dump(fold_out / 'results.json', result)
    report(args, result)


def report(args, rows):
    out = Path(args.output) / f'Fold_{args.fold}'
    data = np.load(Path(args.output) / f'fold_{args.fold}_features.npz')
    labels, logits = data['labels'], data['logits']
    split = partitions(len(labels), args.fold)
    # Apply the same full-task threshold search to the original logits as a control.
    for r in rows:
        if 'baseline_rethresholded' not in r['operational']:
            weak, conf = r['weak_index'], r['conf_index']
            va, te = split['val'], split['test']
            sv, st = logits[va,weak]-logits[va,conf], logits[te,weak]-logits[te,conf]
            bv, bt = logits[va].argmax(1), logits[te].argmax(1)
            choice = operational_threshold(labels[va], bv, sv, weak, conf)
            pred = bt.copy()
            gate = np.isin(bt, [weak, conf])
            if choice['enabled']:
                pred[gate] = np.where(st[gate] >= choice['threshold'], weak, conf)
            r['operational']['baseline_rethresholded'] = {
                'val_choice': choice, 'test': multiclass_metrics(labels[te],pred),
                'test_changed': int((pred != bt).sum())}
            dump(out / f'{weak}_{conf}' / 'results.json', r)
    dump(out / 'results.json', rows)
    csv_rows = []
    sections = []
    colors = ['#333333','#0072b2','#009e73','#d55e00','#8e44ad']
    for r in rows:
        dest = out / f'{r["weak_index"]}_{r["conf_index"]}'
        scores = np.load(dest / 'scores.npz')
        series = [('baseline', scores['baseline_test'])] + [(name, scores[name+'_test']) for name in
                  ['embedding_logistic','spectral_logistic','embedding_spectral_logistic']]
        chosen = r['selected_by_validation_ap']
        if chosen not in [name for name, _ in series]:
            series.append((chosen, scores[chosen+'_test']))
        svg = ['<svg viewBox="0 0 520 330" width="520" role="img" aria-label="Test precision recall curve">',
               '<path d="M45 15V280H500" fill="none" stroke="#888"/>',
               '<text x="230" y="315">Recall</text><text x="5" y="15">Precision</text>',
               '<text x="40" y="300">0</text><text x="490" y="300">1</text>',
               '<text x="20" y="280">0</text><text x="20" y="30">1</text>']
        prior = float(scores['test_labels'].mean())
        svg.append(f'<path d="M45 {280-260*prior:.1f}H500" stroke="#aaa" stroke-dasharray="4,4"/>')
        for (name, s), color in zip(series, colors):
            p, re, _ = precision_recall_curve(scores['test_labels'], s)
            ix = np.unique(np.linspace(0,len(p)-1,min(len(p),500),dtype=int))
            points = ' '.join(f'{45+455*re[i]:.1f},{280-260*p[i]:.1f}' for i in ix)
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.7"/>')
        svg.append('</svg>')
        legend = ' · '.join(f'<span style="color:{c}">{html.escape(n)}</span>' for (n,_),c in zip(series, colors))
        table = ['<table><tr><th>Model</th><th>Validation AP</th><th>Test AP</th><th>Test ROC AUC</th><th>Precision</th><th>Recall</th><th>IoU</th></tr>']
        for name, vals in [('baseline',r['baseline'])]+[(m['name'],m) for m in r['models']]:
            v,t = vals['val'], vals['test']
            table.append('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{100*x:.1f}</td>' for x in
                         [v['ap'],t['ap'],t['roc_auc'],t['precision'],t['recall'],t['iou']])+'</tr>')
            csv_rows.append({'weak':r['weak'],'confounder':r['confounder'],'model':name,
                             'validation_ap':v['ap'], **{'test_'+k:value for k,value in t.items()}})
        table.append('</table>')
        op = r['operational']
        operational_table = ['<h3>Full 20-class check: one pair changed at a time</h3>',
            '<table><tr><th>Rule</th><th>Test overall accuracy</th><th>Test mIoU</th></tr>']
        for label, met in [('Original baseline', op['test_baseline']),
                           ('Baseline with validation-tuned pair threshold', op['baseline_rethresholded']['test']),
                           ('Validation-selected probe, validation-tuned threshold', op['test_rerouted'])]:
            operational_table.append(f'<tr><td>{label}</td><td>{100*met["oa"]:.3f}</td><td>{100*met["miou"]:.3f}</td></tr>')
        operational_table.append('</table>')
        sections.append(f'<section><h2>{html.escape(r["weak"])} vs {html.escape(r["confounder"])}</h2>'
            f'<p>Counts: {html.escape(str(r["counts"]))}. Model selected by validation AP: {r["selected_by_validation_ap"]}.</p>'
            + ''.join(table)+''.join(svg)+'<p>'+legend+f'</p><p>Dashed line: positive-class prevalence ({100*prior:.2f}%). '
            + 'Selected-model AP difference versus baseline, stratified paired bootstrap 95% interval: '
            + html.escape(str([round(100*x,2) for x in r['selected_test_ap_delta_ci95']]))
            + ' percentage points. This conditions on fitted models and does not include training or model-selection uncertainty.</p>'
            + ''.join(operational_table)
            + f'<p>Probe changed {op["test_changed"]} test predictions. The baseline gate includes {op["test_gate_other_class_count"]} parcels whose true class is outside this pair; these errors remain in the full-task evaluation. '
            + 'Both threshold searches may select no change. Gains here are exploratory single-fold results, not a combined multi-pair classifier.</p></section>')
    with open(out / 'metrics.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    document='''<!doctype html><meta charset="utf-8"><title>Frozen feature pairwise probes</title>
<style>body{font:16px/1.5 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#222}table{border-collapse:collapse;font-size:14px}td,th{padding:7px 12px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}section{margin:40px 0;padding-top:10px;border-top:2px solid #ddd}svg{max-width:100%}code{background:#eee}</style>
<h1>Frozen feature pairwise probes — fold '''+str(args.fold)+'''</h1>
<p>Exploratory diagnostic, not a replacement 20-class benchmark. Original seed-1 60/20/20 split. No encoder fitting; evaluation mode, fixed 64-pixel sample, no jitter. Baseline and probes use identical parcels. Validation AP selects regularization/class weighting and model family; pair thresholds maximize validation positive-class IoU. Test data is used for reporting only. The original checkpoint was already selected on this validation split.</p>
<p>Baseline means the pairwise logit difference, with its threshold also tuned on validation. AP is average precision (higher is better); it measures ranking independently of the chosen threshold. Percentages shown below. Pairwise precision depends on the retained two-class prevalence. Results do not establish full-task gains.</p>
<h2>Features and interpretation</h2>
<p>Embedding: 128 values after frozen temporal encoder, before decoder. Bands: all-pixel mean and population standard deviation for 10 spectral bands at each of 24 dates (480 values), plus four geometry features. Standard band order: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12; archive META contains no explicit channel-name file. B5/B6/B7 sample the red-edge reflectance rise; B8/B8A are near infrared; B11/B12 are short-wave infrared. The baseline already receives these bands.</p>
<p>Spectral adds six ratios of parcel means at each date: NDVI=(B8−B4)/(B8+B4); NDRE5/6/7=(B8A−B5/6/7)/(B8A+B5/6/7); NDMI=(B8A−B11)/(B8A+B11); ND_SWIR=(B11−B12)/(B11+B12). NDVI describes vegetation greenness; red-edge contrasts are sensitive to vegetation/chlorophyll-related differences; NDMI is sensitive to canopy water status; the SWIR contrast describes differences in the two SWIR responses influenced by water, dry matter and soil. These are proxies, not direct physical measurements.</p>
<p>Also included: adjacent index changes divided by days between acquisitions, time-weighted seasonal mean (trapezoidal integral divided by duration), minimum, maximum, amplitude, peak days since first acquisition/365, maximum and minimum daily change, and means for Jan–Mar, Apr–Jun, Jul–Aug, Sep–Oct. Indices are computed before standardization; ratios of parcel means are not the same as means of pixelwise ratios. Scaling is fitted on training parcels only. Summaries use all pixels, whereas embeddings use one fixed 64-pixel sample; any benefit can reflect that difference as well as the explicit features.</p>
<p>Full-task check: for the validation-selected pair model, reroute only parcels whose baseline prediction is one of the pair classes. Choose a threshold on full validation mIoU, allowing no change; apply it unchanged on all test classes. Each pair is tested separately, not combined. Third-class mistakes are retained.</p>
<p>Sources: <a href="https://arxiv.org/pdf/1911.07757">PSE+TAE paper</a>; <a href="https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndmi/">NDMI definition</a>; <a href="https://step.esa.int/docs/extra/ATBD_S2ToolBox_L2B_V1.1.pdf">ESA vegetation indices</a>.</p>
'''+''.join(sections)
    (out / 'report.html').write_text(document)
    print('REPORT',out/'report.html',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--moments',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fold',type=int,default=1);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--stage',choices=['all','extract','fit'],default='all')
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1)
    with threadpool_limits(limits=args.threads):
        if args.stage in ['all','extract']:extract(args)
        if args.stage in ['all','fit']:fit(args)


if __name__=='__main__':main()

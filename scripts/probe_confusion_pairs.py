"""Validation-ranked confusion pairs: size-matched moment probes and spectral curves.

prepare reads native NPY headers (not the pixel arrays) to obtain exact counts.
fit trains nested logistic probes on disjoint, size-matched train/val/test sets.
baseline records the original classifier's recall on the same matched parcels.
plot only reads the saved probe scores and spectral-curve records.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.splits import partitions

NAMES = ['Meadow', 'Winter Durum Wheat', 'Spring Cereal', 'Summer Cereal',
         'Winter Cereal', 'Sorgho/millet/moha', 'Cereal', 'Leguminous Fodder',
         'Other Fodder', 'Winter Rapeseed', 'Sunflower', 'Soy', 'Protein Crops',
         'Potato', 'Fruits/Legumes/Flowers', 'Ligneous', 'Orchard', 'Grapevine',
         'Non-agricultural', 'Woodlands']
FEATURES = ['Mean', 'Mean + std', '+ skewness', '+ excess kurtosis', '+ standardized fifth moment']
INDICES = [('NDVI', 6, 2), ('NDRE5', 7, 3), ('NDRE6', 7, 4),
           ('NDRE7', 7, 5), ('NDMI', 7, 8), ('SWIR contrast', 8, 9)]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def miou(cm):
    tp = cm.diagonal()
    return float(np.mean(tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)))


def rank_pairs(cm):
    reference = miou(cm)
    rows = []
    for true in range(len(cm)):
        for pred in range(len(cm)):
            n = int(cm[true, pred])
            if true == pred or not n:
                continue
            corrected = cm.copy()
            corrected[true, pred] = 0
            corrected[true, true] += n
            rows.append(dict(true=true, predicted=pred, true_name=NAMES[true],
                             predicted_name=NAMES[pred], errors=n,
                             oracle_miou_gain_points=100 * (miou(corrected) - reference)))
    return sorted(rows, key=lambda r: (-r['oracle_miou_gain_points'], -r['errors'], r['true'], r['predicted']))


def pixel_count(path):
    with path.open('rb') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f'Unsupported NPY header {version}: {path}')
    assert len(shape) == 3 and shape[:2] == (24, 10), (path, shape)
    return shape[-1]


def prepare(a):
    previous = a.output / 'selection.json'
    if previous.exists():
        record = json.loads(previous.read_text())
        assert record['fold'] == a.fold, 'Use a separate output directory for each fold'
        assert record['baseline_cache_sha256'] == sha(a.baseline_cache), 'Baseline cache changed'
    labels = np.load(a.features / 'targets.npy')
    split = partitions(len(labels), a.fold)
    assert f'fold_{a.fold}_' in a.baseline_cache.name, 'Use the baseline cache from the requested fold'
    with np.load(a.baseline_cache) as f:
        np.testing.assert_array_equal(f['labels'], labels)
        pids = f['pids'].astype(str)
        assert len(np.unique(pids)) == len(pids)
        np.testing.assert_array_equal(pids.astype(np.int64), np.sort(pids.astype(np.int64)))
        if (a.features / 'pids.npy').exists():
            np.testing.assert_array_equal(np.load(a.features / 'pids.npy').astype(str), pids)
        y = labels[split['val']]
        pred = f['logits'][split['val']].argmax(1)
    if (a.output / 'pids.npy').exists():
        np.testing.assert_array_equal(np.load(a.output / 'pids.npy'), pids)
    np.save(a.output / 'pids.npy', pids)
    cm = np.bincount(20 * y + pred, minlength=400).reshape(20, 20)
    ranking = rank_pairs(cm)
    top_count = max(ranking, key=lambda r: (r['errors'], -r['true'], -r['predicted']))
    selection = dict(fold=a.fold, selection_split='validation', baseline_cache_sha256=sha(a.baseline_cache),
                     confusion=cm.tolist(), validation_miou=miou(cm), ranking=ranking,
                     top_count=top_count, top_five=ranking[:5],
                     definition='Hypothetically correct all true→predicted errors for one directed pair; '
                                'leave all other predictions unchanged. Gains are not additive.',
                     diagnostic_caveat='Pair selection uses validation only; fold 1 has already been inspected repeatedly.')
    write(a.output / 'selection.json', selection)
    print('SELECTED', json.dumps(dict(top_count=top_count, top_five=ranking[:5])), flush=True)
    path = a.output / 'pixel_counts.npy'
    if path.exists():
        assert np.load(path).shape == pids.shape, 'Pixel-count cache length changed'
    if not path.exists():
        started = time.time()
        paths = [a.dataset / 'DATA' / f'{pid}.npy' for pid in pids]
        counts = np.empty(len(paths), dtype=np.int32)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for i, count in enumerate(pool.map(pixel_count, paths)):
                counts[i] = count
                if (i + 1) % 20000 == 0:
                    print('HEADERS', i + 1, 'seconds', round(time.time() - started), flush=True)
        np.save(path, counts)
    write(a.output / 'count_provenance.json', dict(pids_sha256=sha(a.output / 'pids.npy'),
                                                 pixel_counts_sha256=sha(path), n=len(pids),
                                                 source='native NPY shape[-1]'))


def exact_match(ids, labels, counts, positive, negative, seed):
    """Balanced, one-to-one exact count matches; scores/features are not inputs."""
    rng = np.random.RandomState(seed)
    groups = [defaultdict(list), defaultdict(list)]
    for cls, group in zip([positive, negative], groups):
        selected = ids[(labels[ids] == cls) & (counts[ids] >= 6)].copy()
        rng.shuffle(selected)
        for i in selected:
            group[int(counts[i])].append(int(i))
    pairs = []
    for size in sorted(groups[0].keys() & groups[1].keys()):
        pairs.extend(zip(groups[0][size], groups[1][size]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def measure(y, score):
    return dict(roc_auc=float(roc_auc_score(y, score)), ap=float(average_precision_score(y, score)))


def fast_auc(y, score):
    n = int(y.sum())
    return float((rankdata(score)[y == 1].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


def difference_interval(y, first, second, seed=20261021, draws=2000):
    """Rows alternate positive/negative; resample the original size-matched pairs."""
    rng = np.random.RandomState(seed)
    n = len(y) // 2
    delta = []
    for _ in range(draws):
        pairs = rng.randint(n, size=n)
        ix = np.column_stack([2 * pairs, 2 * pairs + 1]).ravel()
        delta.append(fast_auc(y[ix], second[ix]) - fast_auc(y[ix], first[ix]))
    return dict(delta=fast_auc(y, second) - fast_auc(y, first),
                ci95=np.quantile(delta, [.025, .975]).tolist(), draws=draws, seed=seed,
                caveat='Conditional on matched pairs and fitted classifiers; not model-selection or spatial uncertainty.')


def dates_from(path):
    calendar = json.loads(path.read_text())
    return calendar['dates'] if 'dates' in calendar else [calendar[str(i)] for i in range(len(calendar))]


def spectral_curves(mom, ids, y, dates, positive, negative):
    means = np.asarray(mom[ids, :, :, 0])
    result = dict(dates=dates, positive=NAMES[positive], negative=NAMES[negative], indices={})
    for name, left, right in INDICES:
        num = means[:, :, left] - means[:, :, right]
        den = means[:, :, left] + means[:, :, right]
        values = np.divide(num, den, out=np.full_like(num, np.nan), where=np.abs(den) > 1e-6)
        result['indices'][name] = {}
        for label, cls in [('positive', 1), ('negative', 0)]:
            q = np.nanpercentile(values[y == cls], [25, 50, 75], axis=0)
            # Preserve missing dates in JSON rather than nonstandard NaN.
            result['indices'][name][label] = dict(
                q25=[float(x) if np.isfinite(x) else None for x in q[0]],
                median=[float(x) if np.isfinite(x) else None for x in q[1]],
                q75=[float(x) if np.isfinite(x) else None for x in q[2]],
                valid=np.isfinite(values[y == cls]).sum(0).tolist())
    return result


def fit_pair(a, positive, negative):
    folder = a.output / f'{positive}_{negative}'
    if (folder / 'metrics.json').exists():
        previous = json.loads((folder / 'metrics.json').read_text())
        assert previous['fold'] == a.fold and previous['seed'] == a.seed, 'Saved protocol differs'
        assert previous['feature_cache_sha256'] == sha(a.features / 'moments.npy'), 'Feature cache changed'
        print('DONE ALREADY', positive, negative, flush=True)
        return
    labels = np.load(a.features / 'targets.npy')
    sizes = np.load(a.output / 'pixel_counts.npy')
    pids = np.load(a.output / 'pids.npy')
    mom = np.load(a.features / 'moments.npy', mmap_mode='r')
    split = partitions(len(labels), a.fold)
    matched = {s: exact_match(ids, labels, sizes, positive, negative, a.seed + j)
               for j, (s, ids) in enumerate(split.items())}
    if min(len(v) for v in matched.values()) < 2:
        raise ValueError(f'Insufficient exact size overlap: {positive}, {negative}')
    ids = {s: v.ravel() for s, v in matched.items()}
    ys = {s: (labels[ix] == positive).astype(int) for s, ix in ids.items()}
    for s, pairs in matched.items():
        assert np.array_equal(sizes[pairs[:, 0]], sizes[pairs[:, 1]])
        assert len(np.unique(pairs)) == pairs.size
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'matched_samples.npz',
                        **{s: v for s, v in matched.items()},
                        **{s + '_pids': pids[v] for s, v in matched.items()})
    data = {s: np.asarray(mom[ix]) for s, ix in ids.items()}
    models, saved_scores = [], {}
    for k in range(1, 6):
        x = {s: values[:, :, :, :k].reshape(len(values), -1) for s, values in data.items()}
        scaler = StandardScaler().fit(x['train'])
        x = {s: scaler.transform(v) for s, v in x.items()}
        trials, best = [], None
        for c in [.01, .1, 1.]:
            model = LogisticRegression(C=c, class_weight='balanced', solver='lbfgs', max_iter=3000, tol=1e-4)
            with threadpool_limits(limits=a.threads), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(x['train'], ys['train'])
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            val = measure(ys['val'], model.decision_function(x['val']))
            trials.append(dict(C=c, converged=converged, validation=val))
            if converged and (best is None or val['ap'] > best[0]):
                best = (val['ap'], c, model)
        if best is None:
            raise RuntimeError(f'No converged classifier: {positive}, {negative}, {k}')
        _, c, model = best
        scores = {s: model.decision_function(v) for s, v in x.items()}
        name = f'moments_1_to_{k}'
        dest = folder / f'{name}.joblib'
        joblib.dump(dict(model=model, scaler=scaler, fold=a.fold, positive=positive, negative=negative,
                         moments=k), dest)
        restored = joblib.load(dest)
        np.testing.assert_array_equal(scores['test'], restored['model'].decision_function(
            restored['scaler'].transform(data['test'][:, :, :, :k].reshape(len(data['test']), -1))))
        models.append(dict(name=name, features=240*k, C=c, trials=trials, checkpoint_sha256=sha(dest),
                           **{s: measure(ys[s], score) for s, score in scores.items()}))
        saved_scores.update({s + '_' + name: v for s, v in scores.items()})
        print('FIT', positive, negative, k, json.dumps(models[-1]), flush=True)
    selected = max(models, key=lambda r: r['val']['ap'])['name']
    report = dict(fold=a.fold, positive=positive, negative=negative,
                  positive_name=NAMES[positive], negative_name=NAMES[negative],
                  protocol='Exact native pixel counts matched separately within train/validation/test; '
                           '>=6 pixels; train-only StandardScaler; C chosen by validation AP; '
                           'nested cumulative raw-band pixel statistics, no encoder features.',
                  seed=a.seed, counts={s: dict(per_class=len(v), n=2*len(v), prevalence=.5) for s, v in matched.items()},
                  pixel_count_roc_auc={s: float(roc_auc_score(ys[s], sizes[ix])) for s, ix in ids.items()},
                  models=models, selected_by_validation_ap=selected,
                  test_std_minus_mean=difference_interval(ys['test'], saved_scores['test_moments_1_to_1'],
                                                         saved_scores['test_moments_1_to_2']),
                  feature_cache_sha256=sha(a.features / 'moments.npy'),
                  matched_sample_sha256=sha(folder / 'matched_samples.npz'))
    np.savez_compressed(folder / 'scores.npz', **saved_scores,
                        **{s + '_indices': v for s, v in ids.items()},
                        **{s + '_labels': v for s, v in ys.items()})
    write(folder / 'curves.json', spectral_curves(mom, ids['test'], ys['test'],
                                                 dates_from(a.calendar), positive, negative))
    write(folder / 'metrics.json', report)


def baseline_pair(a, positive, negative):
    """Keep the original multiclass decision rule on the two-class cohort."""
    folder = a.output / f'{positive}_{negative}'
    selection = json.loads((a.output / 'selection.json').read_text())
    cache_hash = sha(a.baseline_cache)
    assert cache_hash == selection['baseline_cache_sha256']
    report = json.loads((folder / 'metrics.json').read_text())
    assert sha(folder / 'matched_samples.npz') == report['matched_sample_sha256']
    result = dict(fold=a.fold, positive=positive, negative=negative,
                  baseline_cache_sha256=cache_hash,
                  matched_sample_sha256=report['matched_sample_sha256'],
                  rule='Original argmax over all classes, evaluated on the same size-matched '
                       'two-class parcels as the ROC curves. Predictions outside the pair remain errors.',
                  splits={})
    with np.load(a.baseline_cache) as baseline, np.load(folder / 'matched_samples.npz') as matched:
        splits = partitions(len(baseline['labels']), a.fold)
        logits = baseline['logits']
        for split in ['train', 'val', 'test']:
            ids = matched[split].ravel()
            assert np.isin(ids, splits[split]).all()
            np.testing.assert_array_equal(baseline['pids'][ids].astype(str),
                                          matched[split + '_pids'].ravel().astype(str))
            labels = baseline['labels'][ids]
            assert np.isin(labels, [positive, negative]).all()
            predicted = logits[ids].argmax(1)
            n = int((labels == positive).sum())
            tp = int(((labels == positive) & (predicted == positive)).sum())
            fp = int(((labels == negative) & (predicted == positive)).sum())
            result['splits'][split] = dict(positive_count=n, true_positives=tp,
                                           false_positives=fp, recall=tp / n,
                                           false_positive_rate=fp / int((labels == negative).sum()))
    write(folder / 'baseline_recall.json', result)
    print(json.dumps(result, indent=2), flush=True)


def plot_pair(a, positive, negative):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    folder = a.output / f'{positive}_{negative}'
    dest = a.figures / f'{positive}_{negative}'
    dest.mkdir(parents=True, exist_ok=True)
    report = json.loads((folder / 'metrics.json').read_text())
    scores = np.load(folder / 'scores.npz')

    def save(fig, name):
        path = dest / f'{name}.svg'
        fig.savefig(path)
        path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines()) + '\n')
        fig.savefig(folder / f'{name}.png', dpi=120)
        plt.close(fig)

    for mode in ['roc', 'pr']:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
        for ax, split in zip(axes, ['train', 'val', 'test']):
            y = scores[split + '_labels']
            for model, label in zip(report['models'], FEATURES):
                score = scores[split + '_' + model['name']]
                if mode == 'roc':
                    x, v, _ = roc_curve(y, score)
                    ax.plot(x, v, label=label)
                else:
                    v, x, _ = precision_recall_curve(y, score)
                    ax.step(x, v, label=label, where='post')
            if mode == 'roc':
                ax.plot([0, 1], [0, 1], '--', color='#aaa')
            else:
                ax.axhline(y.mean(), ls='--', color='#aaa', label='Sample prevalence')
            ax.set(title={'train': 'Training', 'val': 'Validation', 'test': 'Test'}[split],
                   xlabel='False-positive rate' if mode == 'roc' else 'Recall',
                   ylabel='True-positive rate' if mode == 'roc' else 'Precision', xlim=(0, 1), ylim=(0, 1.02))
            ax.grid(alpha=.15)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{NAMES[positive]} vs {NAMES[negative]}", fontsize=12)
        save(fig, mode)
    curves = json.loads((folder / 'curves.json').read_text())
    dates = np.array([np.datetime64(f'{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}') for d in curves['dates']])
    for names, name in [(['NDVI'], 'ndvi'), ([x[0] for x in INDICES], 'indices'),
                        ([x[0] for x in INDICES if x[0] != 'NDVI'], 'other-indices')]:
        fig, axes = plt.subplots(1 if len(names) == 1 else 2, 1 if len(names) == 1 else 3,
                                 figsize=(7, 4) if len(names) == 1 else (13, 7), layout='constrained')
        for ax, index in zip(np.atleast_1d(axes).flat, names):
            for side, color in [('positive', '#ac4d28'), ('negative', '#176a9c')]:
                curve = curves['indices'][index][side]
                ax.plot(dates, np.array(curve['median'], float), label=curves[side], color=color)
                ax.fill_between(dates, np.array(curve['q25'], float), np.array(curve['q75'], float), color=color, alpha=.15)
            ax.set(title=index, xlabel='Date', ylabel='Index value')
            ax.grid(alpha=.15)
            ax.tick_params(axis='x', rotation=25)
        for ax in list(np.atleast_1d(axes).flat)[len(names):]:
            ax.axis('off')
        np.atleast_1d(axes).flat[0].legend(fontsize=8)
        fig.suptitle(f"{NAMES[positive]} vs {NAMES[negative]}", fontsize=12)
        save(fig, name)


def evaluate_pair(a, positive, negative):
    """Reload the fitted models and verify their scores and recorded metrics."""
    folder = a.output / f'{positive}_{negative}'
    report = json.loads((folder / 'metrics.json').read_text())
    assert report['fold'] == a.fold
    assert sha(a.features / 'moments.npy') == report['feature_cache_sha256']
    assert sha(folder / 'matched_samples.npz') == report['matched_sample_sha256']
    labels = np.load(a.features / 'targets.npy')
    mom = np.load(a.features / 'moments.npy', mmap_mode='r')
    splits = partitions(len(labels), a.fold)
    verified = {}
    with np.load(folder / 'scores.npz') as saved, np.load(folder / 'matched_samples.npz') as matched:
        for entry in report['models']:
            path = folder / f"{entry['name']}.joblib"
            assert sha(path) == entry['checkpoint_sha256']
            fitted = joblib.load(path)
            assert (fitted['fold'], fitted['positive'], fitted['negative']) == (a.fold, positive, negative)
            verified[entry['name']] = {}
            for split in ['train', 'val', 'test']:
                ids = matched[split].ravel()
                assert np.isin(ids, splits[split]).all()
                y = (labels[ids] == positive).astype(int)
                np.testing.assert_array_equal(ids, saved[split + '_indices'])
                np.testing.assert_array_equal(y, saved[split + '_labels'])
                x = np.asarray(mom[ids, :, :, :fitted['moments']]).reshape(len(ids), -1)
                score = fitted['model'].decision_function(fitted['scaler'].transform(x))
                # The fitted estimator uses float32; BLAS/platform rounding can
                # change scores by a few ulps and occasionally reorder near ties.
                np.testing.assert_allclose(score, saved[split + '_' + entry['name']], rtol=1e-5, atol=1e-5)
                metrics = measure(y, score)
                for key, value in metrics.items():
                    # Float32 score ties can move AP in its last decimal places.
                    assert np.isclose(value, entry[split][key], rtol=0, atol=1e-7)
                verified[entry['name']][split] = dict(
                    metrics=metrics,
                    metric_difference={key: value-entry[split][key] for key, value in metrics.items()},
                    max_abs_score_difference=float(np.max(np.abs(score-saved[split+'_'+entry['name']]))))
    write(folder / 'evaluation.json', dict(fold=a.fold, metrics_sha256=sha(folder / 'metrics.json'),
                                           models=verified, status='Saved scores and metrics reproduced within recorded float32 tolerances',
                                           score_tolerance=dict(rtol=1e-5, atol=1e-5), metric_atol=1e-7))
    print('VERIFIED', positive, negative, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'fit', 'evaluate', 'baseline', 'plot'])
    p.add_argument('--features', type=Path, default=Path('results/xgboost'))
    p.add_argument('--dataset', type=Path, default=Path('data/S2-2017-T31TFM-PixelSet'))
    p.add_argument('--baseline-cache', type=Path, default=Path('results/pair_probes/fold_1_features.npz'))
    p.add_argument('--calendar', type=Path, default=Path('data/S2-2017-T31TFM-PixelSet/META/dates.json'))
    p.add_argument('--output', type=Path, default=Path('outputs/confusion_pairs'))
    p.add_argument('--figures', type=Path, default=Path('figures/confusion-pairs'))
    p.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--pair', help='Only one directed pair, e.g. 18_0')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.stage == 'prepare':
        prepare(a)
        return
    selected = json.loads((a.output / 'selection.json').read_text())
    assert selected['fold'] == a.fold, 'Use the selected pairs from the requested fold'
    pairs = list(dict.fromkeys((r['true'], r['predicted']) for r in [selected['top_count']] + selected['top_five']))
    if a.pair:
        pair = tuple(map(int, a.pair.split('_')))
        assert pair in pairs
        pairs = [pair]
    for positive, negative in pairs:
        if a.stage == 'fit':
            fit_pair(a, positive, negative)
        elif a.stage == 'evaluate':
            evaluate_pair(a, positive, negative)
        elif a.stage == 'baseline':
            baseline_pair(a, positive, negative)
        else:
            plot_pair(a, positive, negative)


if __name__ == '__main__':
    main()

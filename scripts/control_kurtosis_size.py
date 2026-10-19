"""Re-evaluate saved kurtosis scores after matching native pixel counts within folds.

No fitting or score-dependent selection. Two fixed analyses: exact counts and
one-to-one matching with max(count)/min(count) <= 1.05. Assignment maximizes the
number of pairs first, then minimizes total absolute log count difference.
Bootstrap intervals condition on the selected pairs and fitted encoders.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def match(rows, ratio):
    pairs = []
    for fold in sorted({r['fold'] for r in rows}):
        a = sorted((r for r in rows if r['fold'] == fold and r['label'] == 0),
                   key=lambda r: (r['pixels'], r['pid']))
        b = sorted((r for r in rows if r['fold'] == fold and r['label'] == 18),
                   key=lambda r: (r['pixels'], r['pid']))
        if not a or not b:
            continue
        distance = np.abs(np.log(np.array([r['pixels'] for r in a])[:, None] /
                                 np.array([r['pixels'] for r in b])[None, :]))
        allowed = distance <= np.log(ratio) + 1e-12
        # One unmatched penalty exceeds the largest possible total valid cost.
        penalty = (len(a) + len(b) + 1) * (1 + np.log(ratio))
        cost = np.full((len(a), len(b) + len(a)), penalty)
        cost[:, :len(b)] = np.where(allowed, distance, 3 * penalty)
        ii, jj = linear_sum_assignment(cost)
        for i, j in zip(ii, jj):
            if j < len(b):
                assert allowed[i, j]
                pairs.append({'fold': fold, 'meadow': a[i], 'non_agricultural': b[j]})
    return pairs


def scores(rows):
    y = np.array([r['label'] == 0 for r in rows])
    k = np.array([r['kurtosis'] for r in rows])
    n = np.array([r['pixels'] for r in rows])
    return dict(n=len(rows), per_class=int(y.sum()),
                kurtosis_roc_auc=float(roc_auc_score(y, k)),
                kurtosis_ap=float(average_precision_score(y, k)),
                pixel_count_roc_auc=float(roc_auc_score(y, n)),
                pixel_count_ap=float(average_precision_score(y, n)))


def flatten(pairs):
    return [r for p in pairs for r in [p['meadow'], p['non_agricultural']]]


def paired_metrics(pairs):
    result = scores(flatten(pairs))
    delta = np.array([p['meadow']['kurtosis'] - p['non_agricultural']['kurtosis'] for p in pairs])
    result['within_pair_concordance'] = float(np.mean((delta > 0) + .5 * (delta == 0)))
    result['max_pixel_count_ratio'] = float(max(
        max(p['meadow']['pixels'], p['non_agricultural']['pixels']) /
        min(p['meadow']['pixels'], p['non_agricultural']['pixels']) for p in pairs))
    return result


def intervals(pairs, draws, seed):
    rng = np.random.RandomState(seed)
    folds = sorted({p['fold'] for p in pairs})
    grouped = [[p for p in pairs if p['fold'] == f] for f in folds]
    values = {k: [] for k in ['kurtosis_roc_auc', 'kurtosis_ap', 'within_pair_concordance']}
    for _ in range(draws):
        sampled = [group[i] for group in grouped for i in rng.randint(len(group), size=len(group))]
        metrics = paired_metrics(sampled)
        for k in values:
            values[k].append(metrics[k])
    return {k: np.quantile(v, [.025, .975]).tolist() for k, v in values.items()}


def plot(rows, results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for ax, (title, selected) in zip(axes, [
        ('All sampled parcels', rows),
        ('Exact pixel-count matches', flatten(results['exact']['pairs'])),
        ('Pixel counts within 5%', flatten(results['within_5_percent']['pairs']))]):
        y = [r['label'] == 0 for r in selected]
        for key, label, color in [('kurtosis', 'Excess kurtosis', '#176a9c'),
                                  ('pixels', 'Native pixel count', '#b55b26')]:
            x, v, _ = roc_curve(y, [r[key] for r in selected])
            ax.plot(x, v, label=label, color=color)
        ax.plot([0, 1], [0, 1], '--', color='#999')
        ax.set(title=title, xlabel='False-positive rate', ylabel='True-positive rate',
               xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.savefig(output)
    fig.savefig(output.with_suffix('.png'), dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('records/kurtosis_remeasurement.json'))
    p.add_argument('--output', type=Path, default=Path('records/kurtosis_size_control.json'))
    p.add_argument('--figure', type=Path, default=Path('figures/kurtosis/size-control.svg'))
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    rows = json.loads(a.input.read_text())['records']
    assert len({r['pid'] for r in rows}) == len(rows), 'Repeated parcel'
    result = dict(protocol=dict(
        input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        positive='Meadow', negative='Non-agricultural',
        match_variables=['fold', 'native_pixel_count'],
        matching='one-to-one without replacement; maximum cardinality then minimum absolute log-count distance',
        score_direction='larger kurtosis predicts Meadow; fixed before matching',
        bootstrap=dict(draws=a.bootstrap, seed=20261020, resampling='matched pairs within each fold',
                       interval='percentile 95%; conditions on chosen pairs and fitted encoders'),
        limitation='Descriptive analysis of previously inspected test scores; overlapping size support only. '
                   'No classifier fitting, no area adjustment beyond native pixel count, no equal-pixel resampling.'),
        unmatched=scores(rows))
    for name, ratio in [('exact', 1.0), ('within_5_percent', 1.05)]:
        pairs = match(rows, ratio)
        result[name] = dict(pairs=pairs, metrics=paired_metrics(pairs),
                            ci95=intervals(pairs, a.bootstrap, 20261020),
                            per_fold=[dict(fold=f, **paired_metrics([p for p in pairs if p['fold'] == f]))
                                      for f in sorted({p['fold'] for p in pairs})])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    a.figure.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, result, a.figure)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ['metrics', 'ci95', 'per_fold']}
                      if k in ['exact', 'within_5_percent'] else v
                      for k, v in result.items() if k != 'protocol'}, indent=2))


if __name__ == '__main__':
    main()

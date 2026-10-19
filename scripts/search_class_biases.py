"""Multistart validation mIoU search with validation accuracy >= baseline."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_pr_explorer import NAMES, metrics, sha
from scripts.sweep_class_biases import evaluate


def grid_sweep(logits, labels, biases, grid):
    """Evaluate every coordinate/grid setting using cumulative prediction changes.

    Each parcel switches once, from its strongest rival to the adjusted class.
    Binary search uses actual float additions and numpy's first-class tie rule.
    """
    grid = np.asarray(grid, dtype=float)
    if len(grid) == 0 or not np.all(np.diff(grid) > 0):
        raise ValueError('Grid must be nonempty and strictly increasing.')
    n, classes = logits.shape
    support = np.bincount(labels, minlength=classes)
    adjusted = logits + biases
    rows = []
    for c in range(classes):
        other = adjusted.copy()
        other[:, c] = -np.inf
        rival = other.argmax(1)
        rival_score = other[np.arange(n), rival]
        low, high = np.zeros(n, dtype=int), np.full(n, len(grid), dtype=int)
        while np.any(low < high):
            mid = (low + high) // 2
            score = logits[:, c] + grid[np.minimum(mid, len(grid) - 1)]
            wins = (score > rival_score) | ((score == rival_score) & (c < rival))
            active = low < high
            high = np.where(active & wins, mid, high)
            low = np.where(active & ~wins, mid + 1, low)
        first = low
        switched = np.cumsum(np.bincount(first, minlength=len(grid) + 1))[:-1]

        def events(mask, category):
            counts = np.bincount(first[mask] * classes + category[mask],
                                 minlength=(len(grid) + 1) * classes)
            return counts.reshape(-1, classes).cumsum(0)[:-1]

        predicted = np.bincount(rival, minlength=classes) - events(np.ones(n, bool), rival)
        predicted[:, c] += switched
        right = rival == labels
        tp = np.bincount(labels[right], minlength=classes) - events(right, labels)
        tp[:, c] += np.cumsum(np.bincount(first[labels == c],
                                        minlength=len(grid) + 1))[:-1]
        correct = tp.sum(1)
        miou = (tp / np.maximum(support + predicted - tp, 1)).mean(1)
        recall = (tp / np.maximum(support, 1)).mean(1)
        for j, bias in enumerate(grid):
            rows.append(dict(class_id=c, class_name=NAMES[c], bias=float(bias),
                             miou=float(miou[j]), oa=float(correct[j] / n),
                             correct=int(correct[j]), mean_recall=float(recall[j])))
    return rows


def optimize(logits, labels, start, minimum_correct, max_updates=40):
    biases = np.asarray(start, dtype=float).copy()
    current = metrics(labels, (logits + biases).argmax(1))['miou']
    trace, phases = [], []
    for step_size in [.1, .01]:
        converged = False
        for step in range(max_updates):
            grid = np.unique(np.r_[np.round(np.arange(-3, 3 + step_size / 2, step_size), 10),
                                   biases])
            rows = grid_sweep(logits, labels, biases, grid)
            feasible = [row for row in rows if row['correct'] >= minimum_correct]
            winner = max(feasible, key=lambda r: (
                r['miou'], -abs(r['bias'] - biases[r['class_id']]),
                -r['class_id'], -abs(r['bias'])))
            if winner['miou'] <= current + 1e-12:
                converged = True
                break
            biases[winner['class_id']] = winner['bias']
            current = winner['miou']
            trace.append(dict(step_size=step_size, update=step + 1, **winner))
        phases.append(dict(step_size=step_size, converged=converged,
                           accepted_updates=sum(r['step_size'] == step_size for r in trace)))
    actual = metrics(labels, (logits + biases).argmax(1))
    assert np.trace(actual['confusion']) >= minimum_correct
    assert abs(actual['miou'] - current) < 1e-12
    return dict(start=list(start), biases=biases.tolist(), trace=trace,
                phases=phases, validation=actual)


def fit(a):
    provenance = json.loads((a.data / 'provenance.json').read_text())
    path = a.data / 'validation_predictions.npz'
    assert sha(path) == provenance['predictions_sha256']
    with np.load(path) as d:
        assert str(d['split']) == 'val' and int(d['fold']) == provenance['fold']
        logits, labels = d['logits'].astype(float), d['labels'].astype(int)
    previous = json.loads(a.previous.read_text())
    assert previous['validation_predictions_sha256'] == sha(path)
    center = np.array(previous['biases']['greedy'])
    baseline = metrics(labels, logits.argmax(1))
    minimum_correct = int(np.trace(baseline['confusion']))
    assert np.sum((logits + center).argmax(1) == labels) >= minimum_correct
    rng = np.random.RandomState(20261020)
    starts = [np.zeros(logits.shape[1]), center.copy()]
    for _ in range(6):
        delta = rng.normal(0, .3, logits.shape[1])
        trial = np.clip(center + delta, -3, 3)
        while np.sum((logits + trial).argmax(1) == labels) < minimum_correct:
            delta *= .5
            trial = np.clip(center + delta, -3, 3)
        starts.append(trial)
    record = dict(
        fold=provenance['fold'],
        baseline_cache_sha256=provenance['baseline_cache_sha256'],
        validation_predictions_sha256=sha(path),
        initial_search_sha256=sha(a.previous),
        protocol=dict(
            objective='Maximize full multiclass validation mIoU',
            constraint='Validation number correct >= unchanged baseline',
            minimum_correct=minimum_correct, bounds=[-3, 3], grid_steps=[.1, .01],
            max_updates_per_phase=40, seed=20261020,
            starts='Zero; previous greedy result; six Gaussian perturbations of it with '
                   'sigma 0.3, halved until feasible. All clipped to bounds.',
            search='Best-coordinate ascent at both resolutions for each start. Current biases '
                   'also included as candidates. Select highest validation mIoU; ties prefer '
                   'higher OA, then smaller squared bias norm, then earlier start.',
            tie_rule='First class, matching numpy argmax',
            caveat='Repeatedly inspected validation and test fold; exploratory. '
                   'No test scores used in this search. The OA constraint is validation-only.'),
        runs=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Save the fixed protocol before any run or test evaluation.
    a.output.write_text(json.dumps(record, indent=2) + '\n')
    for i, start in enumerate(starts):
        result = optimize(logits, labels, start, minimum_correct)
        record['runs'].append(result)
        a.output.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(start=i, validation={key: result['validation'][key]
                         for key in ['miou', 'oa', 'mean_recall']}, phases=result['phases'])),
              flush=True)
    best = max(range(len(record['runs'])), key=lambda i: (
        record['runs'][i]['validation']['miou'],
        record['runs'][i]['validation']['oa'],
        -np.square(record['runs'][i]['biases']).sum(), -i))
    selected = record['runs'][best]
    record.update(selected_start=best,
                  biases=dict(baseline=starts[0].tolist(), selected=selected['biases']),
                  validation=dict(baseline=baseline, selected=selected['validation']))
    a.output.write_text(json.dumps(record, indent=2) + '\n')
    print('FROZEN', a.output, 'selected start', best, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['fit', 'evaluate'])
    p.add_argument('--data', type=Path, default=Path('outputs/pr_explorer'))
    p.add_argument('--previous', type=Path, default=Path('records/class_bias_search.json'))
    p.add_argument('--baseline-cache', type=Path,
                   default=Path('results/pair_probes/fold_1_features.npz'))
    p.add_argument('--output', type=Path, default=Path('records/class_bias_search_extended.json'))
    a = p.parse_args()
    (fit if a.stage == 'fit' else evaluate)(a)


if __name__ == '__main__':
    main()

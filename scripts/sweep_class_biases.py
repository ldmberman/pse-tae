"""Validation-only class-bias search; optional frozen-rule test evaluation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_pr_explorer import NAMES, metrics, sha
from learning.splits import partitions


def sweep(logits, labels, biases, grid):
    adjusted = logits + biases
    rows = []
    for c in range(logits.shape[1]):
        other = adjusted.copy()
        other[:, c] = -np.inf
        rival = other.argmax(1)
        rival_score = other[np.arange(len(labels)), rival]
        for bias in grid:
            score = logits[:, c] + bias
            wins = (score > rival_score) | ((score == rival_score) & (c < rival))
            predicted = np.where(wins, c, rival)
            m = metrics(labels, predicted)
            rows.append(dict(class_id=c, class_name=NAMES[c], bias=float(bias),
                             miou=m['miou'], oa=m['oa'], mean_recall=m['mean_recall']))
    return rows


def choose(rows, biases):
    # Deterministic ties prefer the smaller change.
    return max(rows, key=lambda r: (r['miou'], -abs(r['bias'] - biases[r['class_id']]),
                                    -r['class_id'], -abs(r['bias'])))


def fit(a):
    provenance = json.loads((a.data / 'provenance.json').read_text())
    path = a.data / 'validation_predictions.npz'
    assert sha(path) == provenance['predictions_sha256']
    with np.load(path) as d:
        assert str(d['split']) == 'val'
        logits, labels = d['logits'].astype(float), d['labels'].astype(int)
    grid = np.round(np.linspace(-3, 3, 61), 10)
    biases = np.zeros(20)
    baseline = metrics(labels, logits.argmax(1))
    original = sweep(logits, labels, biases, grid)
    winner = choose(original, biases)
    single = biases.copy()
    if winner['miou'] > baseline['miou'] + 1e-12:
        single[winner['class_id']] = winner['bias']
    trace = []
    current = baseline['miou']
    for step in range(5):
        rows = original if step == 0 else sweep(logits, labels, biases, grid)
        winner = choose(rows, biases)
        if winner['miou'] <= current + 1e-12:
            break
        biases[winner['class_id']] = winner['bias']
        current = winner['miou']
        trace.append(dict(step=step + 1, **winner))
        print('ACCEPT', trace[-1], flush=True)
    settings = dict(baseline=np.zeros(20).tolist(), best_single=single.tolist(),
                    greedy=biases.tolist())
    record = dict(fold=provenance['fold'], baseline_cache_sha256=provenance['baseline_cache_sha256'],
                  validation_predictions_sha256=provenance['predictions_sha256'],
                  protocol=dict(grid=grid.tolist(), objective='Full multiclass validation mIoU',
                                max_greedy_updates=5, tie_rule='First class, matching numpy argmax',
                                search='First vary one class at a time from zero. Then accept the best '
                                       'single-coordinate improvement, recomputing all candidates each round. '
                                       'This is not an exhaustive joint search.',
                                caveat='Repeatedly inspected fold; exploratory. Test is not used in selection.'),
                  single_class_sweeps=original, greedy_trace=trace, biases=settings,
                  validation={name: metrics(labels, (logits + bias).argmax(1))
                              for name, bias in settings.items()})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=2) + '\n')
    print('SAVED', a.output)


def evaluate(a):
    record = json.loads(a.output.read_text())
    assert sha(a.baseline_cache) == record['baseline_cache_sha256']
    with np.load(a.baseline_cache) as data:
        ix = partitions(len(data['labels']), record['fold'])['test']
        logits, labels = data['logits'][ix].astype(float), data['labels'][ix].astype(int)
    record['test'] = {name: metrics(labels, (logits + bias).argmax(1))
                      for name, bias in record['biases'].items()}
    record['test_evaluation'] = 'Frozen validation-selected settings; no test tuning.'
    a.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({name: {k: value[k] for k in ['miou', 'oa', 'mean_recall']}
                      for name, value in record['test'].items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['fit', 'evaluate'])
    p.add_argument('--data', type=Path, default=Path('outputs/pr_explorer'))
    p.add_argument('--baseline-cache', type=Path, default=Path('results/pair_probes/fold_1_features.npz'))
    p.add_argument('--output', type=Path, default=Path('records/class_bias_search.json'))
    a = p.parse_args()
    {'fit': fit, 'evaluate': evaluate}[a.stage](a)


if __name__ == '__main__':
    main()

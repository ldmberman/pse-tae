"""CPU multiclass probes; training, selection, and test partitions stay separate."""
import os
if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from frozen_pair_probes import NAMES, partitions, dump


def metrics(y, pred):
    cm = np.bincount(y * 20 + pred, minlength=400).reshape(20, 20)
    tp = cm.diagonal()
    recall = tp / np.maximum(cm.sum(1), 1)
    precision = tp / np.maximum(cm.sum(0), 1)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    return dict(oa=float(tp.sum() / cm.sum()), miou=float(iou.mean()),
                mean_recall=float(recall.mean()), iou=iou.tolist(),
                precision=precision.tolist(), recall=recall.tolist(), confusion=cm.tolist())


def miou_metric(pred, data):
    return 'miou', metrics(data.get_label().astype(int), pred.argmax(1))['miou']


def main(args):
    out = Path(args.output) / f'Fold_{args.fold}'
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    data = np.load(cache / f'fold_{args.fold}_features.npz')
    spectral = np.load(cache / 'spectral_features.npz')
    assert np.array_equal(data['pids'], spectral['pids'])
    y = data['labels'].astype(int)
    split = partitions(len(y), args.fold)
    tr, va, te = [split[k] for k in ('train', 'val', 'test')]
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    counts = np.bincount(y[tr], minlength=20)
    weights = 1 / np.sqrt(counts)
    weights /= np.mean(weights[y[tr]])
    provenance = dict(fold=args.fold, classes=NAMES, counts_train=counts.tolist(),
                      sklearn=sklearn.__version__, xgboost=xgb.__version__,
                      selection='Validation mIoU only, independently within each input/family/weight group',
                      extraction=json.loads((cache / f'fold_{args.fold}_extraction.json').read_text()),
                      linear_C=[0.01, 0.1], tree_depth=[3, 5], tree_rounds=300,
                      weights=weights.tolist(), threads=args.threads)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        assert all(previous[k] == v for k, v in provenance.items() if k != 'threads'), 'Cached protocol mismatch'
    else:
        temporary = out / f'protocol.{os.getpid()}.json'
        dump(temporary, provenance)
        os.replace(temporary, protocol_path)
    matrices = dict(embedding=data['embeddings'], spectral=spectral['spectral'])
    matrices['combined'] = np.concatenate(list(matrices.values()), axis=1)
    records = []
    for features, matrix in matrices.items():
        if features not in args.features:
            continue
        assert np.isfinite(matrix).all()
        scaler = StandardScaler().fit(matrix[tr])
        xtr, xva = scaler.transform(matrix[tr]), scaler.transform(matrix[va])
        for family in ['logistic', 'xgb']:
            for weighting in ['natural', 'sqrt']:
                group = f'{features}_{family}_{weighting}'
                sw = None if weighting == 'natural' else weights[y[tr]]
                best = None
                for value in ([0.01, 0.1] if family == 'logistic' else [3, 5]):
                    name = f'{group}_{value}'
                    path = out / (name + '.json')
                    model_path = out / (name + '.joblib')
                    if path.exists() and model_path.exists():
                        record = json.loads(path.read_text())
                        print('CACHED', name, record['val']['miou'], flush=True)
                    else:
                        start = time.time()
                        print('START', name, flush=True)
                        if family == 'logistic':
                            model = LogisticRegression(C=value, solver='lbfgs', max_iter=600,
                                                       tol=1e-4, random_state=1)
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always', ConvergenceWarning)
                                model.fit(xtr, y[tr], sample_weight=sw)
                            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                            pred = model.predict(xva)
                            detail = dict(C=value, converged=converged,
                                          iterations=int(model.n_iter_.max()))
                            artifact = dict(model=model, scaler=scaler)
                        else:
                            dtr = xgb.DMatrix(matrix[tr], label=y[tr], weight=sw)
                            dva = xgb.DMatrix(matrix[va], label=y[va])
                            params = dict(objective='multi:softprob', num_class=20,
                                          tree_method='hist', max_depth=value, eta=0.07,
                                          min_child_weight=5, reg_lambda=10,
                                          subsample=0.9, colsample_bytree=0.9,
                                          nthread=args.threads, seed=1, disable_default_eval_metric=1)
                            model = xgb.train(params, dtr, num_boost_round=300,
                                              evals=[(dva, 'validation')],
                                              custom_metric=miou_metric, maximize=True,
                                              early_stopping_rounds=40, verbose_eval=50)
                            pred = model.predict(dva, iteration_range=(0, model.best_iteration + 1)).argmax(1)
                            detail = dict(depth=value, best_round=model.best_iteration + 1)
                            artifact = dict(model=model, scaler=None)
                            del dtr, dva
                        record = dict(name=name, group=group, features=features, family=family,
                                      weighting=weighting, detail=detail, val=metrics(y[va], pred),
                                      seconds=time.time() - start)
                        joblib.dump(artifact, model_path)
                        dump(path, record)
                        print('DONE', name, 'val', record['val']['oa'], record['val']['miou'],
                              'seconds', round(record['seconds']), detail, flush=True)
                    records.append(record)
                    if best is None or record['val']['miou'] > best['val']['miou']:
                        best = record
                dump(out / (group + '_selection.json'), best)
        del xtr, xva
    if args.fit_only:
        print('FITTING COMPLETE', args.features, flush=True)
        return
    assert set(args.features) == set(matrices), 'Final selection requires every feature set'
    # Freeze all selections before evaluating any test predictions.
    selected = []
    for group in sorted(set(r['group'] for r in records)):
        selected.append(max((r for r in records if r['group'] == group), key=lambda r: r['val']['miou']))
    winner = max(selected, key=lambda r: r['val']['miou'])['name']
    dump(out / 'selection.json', dict(winner=winner, models=[r['name'] for r in selected]))
    baseline = dict(name='original_decoder', group='original_decoder', features='embedding',
                    family='original_decoder', weighting='original', detail={},
                    val=metrics(y[va], data['logits'][va].argmax(1)),
                    test=metrics(y[te], data['logits'][te].argmax(1)))
    predictions = {'labels': y[te], 'indices': te, 'original_decoder': data['logits'][te].argmax(1)}
    scores = {}
    for s, indices in [('val', va), ('test', te)]:
        logits = data['logits'][indices]
        probability = np.exp(logits - logits.max(1, keepdims=True))
        probability /= probability.sum(1, keepdims=True)
        scores[s] = {'labels': y[indices], 'indices': indices,
                     'pids': data['pids'][indices], 'original_decoder': probability}
    for r in selected:
        obj = joblib.load(out / (r['name'] + '.joblib'))
        for s, indices in [('val', va), ('test', te)]:
            matrix = matrices[r['features']][indices]
            if r['family'] == 'logistic':
                assert np.array_equal(obj['model'].classes_, np.arange(20))
                probability = obj['model'].predict_proba(obj['scaler'].transform(matrix))
            else:
                probability = obj['model'].predict(xgb.DMatrix(matrix),
                                                  iteration_range=(0, r['detail']['best_round']))
            pred = probability.argmax(1)
            scores[s][r['name']] = probability.astype(np.float32)
            if s == 'val':
                assert np.isclose(metrics(y[indices], pred)['miou'], r['val']['miou'], atol=1e-12)
            else:
                r['test'] = metrics(y[indices], pred)
                predictions[r['name']] = pred
    np.savez_compressed(out / 'test_predictions.npz', **predictions)
    for s, arrays in scores.items():
        np.savez_compressed(out / f'{s}_scores.npz', **arrays)
    results = [baseline] + selected
    dump(out / 'results.json', dict(winner=winner, models=results))
    rows = [dict(model=r['name'], split=s, **{k:r[s][k] for k in ['oa','miou','mean_recall']})
            for r in results for s in ['val','test']]
    with open(out / 'metrics.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    rows = [dict(model=r['name'], split=s, class_name=n, **{k:r[s][k][i] for k in ['iou','precision','recall']})
            for r in results for s in ['val','test'] for i,n in enumerate(NAMES)]
    with open(out / 'per_class.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print('COMPLETE', out, 'validation_selected', winner, flush=True)
    for r in results:
        print(r['name'], 'VAL', r['val']['miou'], 'TEST', r['test']['oa'], r['test']['miou'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', default='results/pair_probes')
    p.add_argument('--output', default='results/multiclass_probes')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--features', nargs='+', choices=['embedding', 'spectral', 'combined'],
                   default=['embedding', 'spectral', 'combined'])
    p.add_argument('--fit-only', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=args.threads):
        main(args)

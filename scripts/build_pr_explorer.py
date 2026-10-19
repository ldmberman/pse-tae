"""Export fold-matched logits and build an offline multiclass PR slider figure."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.splits import partitions

NAMES = ['Meadow', 'Winter Durum Wheat', 'Spring Cereal', 'Summer Cereal',
         'Winter Cereal', 'Sorgho/millet/moha', 'Cereal', 'Leguminous Fodder',
         'Other Fodder', 'Winter Rapeseed', 'Sunflower', 'Soy', 'Protein Crops',
         'Potato', 'Fruits/Legumes/Flowers', 'Ligneous', 'Orchard', 'Grapevine',
         'Non-agricultural', 'Woodlands']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def metrics(labels, predicted):
    cm = np.bincount(labels * 20 + predicted, minlength=400).reshape(20, 20)
    tp = cm.diagonal()
    support, counts = cm.sum(1), cm.sum(0)
    precision = tp / np.maximum(counts, 1)
    recall = tp / np.maximum(support, 1)
    iou = tp / np.maximum(support + counts - tp, 1)
    return dict(oa=float(tp.sum() / cm.sum()), miou=float(iou.mean()),
                mean_recall=float(recall.mean()), mean_precision=float(precision.mean()),
                precision=precision.tolist(), recall=recall.tolist(), iou=iou.tolist(),
                support=support.tolist(), predicted=counts.tolist(), confusion=cm.tolist())


def extract(a):
    assert f'fold_{a.fold}_' in a.baseline_cache.name
    source_hash = sha(a.baseline_cache)
    extraction = json.loads(a.baseline_cache.with_name(
        f'fold_{a.fold}_extraction.json').read_text())
    a.data.mkdir(parents=True, exist_ok=True)
    with np.load(a.baseline_cache) as data:
        total = len(data['labels'])
        splits = partitions(total, a.fold)
        for split, stem in [('val', 'validation'), ('test', 'test')]:
            indices = splits[split]
            logits = data['logits'][indices].astype('<f4')
            labels = data['labels'][indices].astype(np.uint8)
            assert logits.shape == (len(labels), len(NAMES)) and np.isfinite(logits).all()
            assert np.all(labels < len(NAMES))
            dest = a.data / f'{stem}_predictions.npz'
            np.savez_compressed(dest, logits=logits, labels=labels, indices=indices,
                                fold=a.fold, split=split)
            record = dict(fold=a.fold, split=split, n=len(labels), total_parcels=total, classes=NAMES,
                          baseline_cache_sha256=source_hash, predictions_sha256=sha(dest),
                          baseline_checkpoint_sha256=extraction['checkpoint_sha256'],
                          baseline=metrics(labels.astype(int), logits.argmax(1)),
                          rule='argmax(original logit + class bias); no training or probability recalibration',
                          selection_note='Bias tuning uses validation; test evaluation requires an explicit button.',
                          zero_prediction_precision='Plot as zero when a class has no predicted parcels.')
            filename = 'provenance.json' if split == 'val' else 'test_provenance.json'
            (a.data / filename).write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps({k: record[k] for k in ['fold', 'split', 'n', 'predictions_sha256']}))


def test_payload(folder, validation):
    record = json.loads((folder / 'test_provenance.json').read_text())
    assert record['split'] == 'test'
    for key in ['fold', 'classes', 'baseline_cache_sha256', 'baseline_checkpoint_sha256']:
        assert record[key] == validation[key], f'Test/validation mismatch: {key}'
    path = folder / 'test_predictions.npz'
    assert sha(path) == record['predictions_sha256']
    splits = partitions(record['total_parcels'], record['fold'])
    with np.load(folder / 'validation_predictions.npz') as data:
        assert np.array_equal(data['indices'], splits['val'])
    with np.load(path) as data:
        assert str(data['split']) == 'test' and int(data['fold']) == record['fold']
        assert np.array_equal(data['indices'], splits['test'])
        key = 'probabilities' if 'probabilities' in data else 'logits'
        scores, labels = data[key].astype('<f4'), data['labels'].astype(np.uint8)
    assert scores.shape == (record['n'], len(NAMES)) and labels.shape == (record['n'],)
    assert np.isfinite(scores).all() and np.all(labels < len(NAMES))
    if key == 'probabilities':
        assert np.all(scores > 0)
        np.testing.assert_allclose(scores.sum(1), 1, rtol=0, atol=1e-6)
    assert metrics(labels.astype(int), scores.argmax(1)) == record['baseline']
    return dict(provenance=record, labels=base64.b64encode(labels.tobytes()).decode(),
                **{key: base64.b64encode(scores.tobytes()).decode()})


def extract_test(a):
    validation = json.loads((a.data / 'provenance.json').read_text())
    protocol = json.loads((a.scores / 'protocol.json').read_text())
    assert protocol['fold'] == validation['fold'] == a.fold
    assert protocol['extraction']['checkpoint_sha256'] == validation['baseline_checkpoint_sha256']
    assert [n.lower() for n in protocol['classes']] == [n.lower() for n in validation['classes']]
    splits = partitions(protocol['extraction']['n'], a.fold)
    with np.load(a.data / 'validation_predictions.npz') as data, np.load(a.scores / 'val_scores.npz') as saved:
        assert sha(a.data / 'validation_predictions.npz') == validation['predictions_sha256']
        assert np.array_equal(data['indices'], splits['val'])
        for key in ['indices', 'labels']:
            assert np.array_equal(data[key], saved[key])
        logits = data['logits'].astype(float)
        probabilities = np.exp(logits - logits.max(1, keepdims=True))
        probabilities /= probabilities.sum(1, keepdims=True)
        np.testing.assert_allclose(probabilities, saved['original_decoder'], rtol=1e-5, atol=5e-7)
    source = a.scores / 'test_scores.npz'
    with np.load(source) as saved:
        assert np.array_equal(saved['indices'], splits['test'])
        probabilities = saved['original_decoder'].astype('<f4')
        labels = saved['labels'].astype(np.uint8)
    assert probabilities.shape == (len(labels), len(NAMES)) and np.all(probabilities > 0)
    assert np.isfinite(probabilities).all() and np.all(labels < len(NAMES))
    np.testing.assert_allclose(probabilities.sum(1), 1, rtol=0, atol=1e-6)
    dest = a.data / 'test_predictions.npz'
    np.savez_compressed(dest, probabilities=probabilities, labels=labels,
                        indices=splits['test'], fold=a.fold, split='test')
    record = dict(fold=a.fold, split='test', n=len(labels),
                  total_parcels=protocol['extraction']['n'], classes=NAMES,
                  baseline_cache_sha256=validation['baseline_cache_sha256'],
                  baseline_checkpoint_sha256=validation['baseline_checkpoint_sha256'],
                  predictions_sha256=sha(dest), source=str(source), source_sha256=sha(source),
                  protocol_sha256=sha(a.scores / 'protocol.json'),
                  score_representation='Log of archived float32 baseline probabilities; equivalent to logits up to a parcel-wise constant and archived rounding.',
                  baseline=metrics(labels.astype(int), probabilities.argmax(1)))
    (a.data / 'test_provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(dest)


def build(a):
    from plotly.offline import get_plotlyjs
    record = json.loads((a.data / 'provenance.json').read_text())
    path = a.data / 'validation_predictions.npz'
    assert sha(path) == record['predictions_sha256']
    assert record['split'] == 'val' and record['fold'] == a.fold
    with np.load(path) as data:
        assert str(data['split']) == 'val' and int(data['fold']) == a.fold
        logits = data['logits'].astype('<f4')
        labels = data['labels'].astype(np.uint8)
        actual = metrics(labels.astype(int), logits.argmax(1))
    assert actual == record['baseline']
    payload = dict(provenance=record, logits=base64.b64encode(logits.tobytes()).decode(),
                   labels=base64.b64encode(labels.tobytes()).decode())
    search_path = a.bias_search
    if search_path is None and a.fold == 1:
        search_path = Path('records/class_bias_search_extended.json')
    if search_path is not None and search_path.is_file():
        search = json.loads(search_path.read_text())
        assert search['validation_predictions_sha256'] == record['predictions_sha256']
        biases = search['biases']['selected']
        measured = metrics(labels.astype(int), (logits.astype(float) + biases).argmax(1))
        assert measured == search['validation']['selected']
        payload['automatic'] = dict(biases=biases, validation=measured,
                                    source_sha256=sha(search_path))
    payload['test'] = test_payload(a.data, record)
    record['test'] = payload['test']['provenance']
    record['selection_note'] = 'Bias tuning uses validation; test evaluation requires an explicit button.'
    root = Path(__file__).resolve().parent
    template = (root / 'pr_explorer.html').read_text()
    html = template.replace('__PLOTLY__', get_plotlyjs())
    html = html.replace('__PAYLOAD__', json.dumps(payload).replace('</', '<\\/'))
    html = html.replace('__SCRIPT__', (root / 'pr_explorer.js').read_text())
    assert chr(0x2019) not in template and chr(0x2014) not in template
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(line.rstrip() for line in html.splitlines()) + '\n')
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(record, indent=2) + '\n')
    print(a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['extract', 'extract-test', 'build'])
    p.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    p.add_argument('--baseline-cache', type=Path, default=Path('results/pair_probes/fold_1_features.npz'))
    p.add_argument('--data', type=Path, default=Path('outputs/pr_explorer'))
    p.add_argument('--output', type=Path, default=Path('figures/multiclass-pr-explorer.html'))
    p.add_argument('--record', type=Path, default=Path('records/pr_explorer.json'))
    p.add_argument('--bias-search', type=Path,
                   help='Optional search record; fold 1 defaults to its extended search when available.')
    p.add_argument('--scores', type=Path, default=Path('results/multiclass_probes/Fold_1'))
    a = p.parse_args()
    {'extract': extract, 'extract-test': extract_test, 'build': build}[a.stage](a)


if __name__ == '__main__':
    main()

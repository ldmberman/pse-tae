"""Validation-only complementary errors and a small probability-mixture sweep.

Input files must contain matching parcel indices and labels. This script never
loads test predictions. Selection and measurement share validation data, so the
reported best mixture is exploratory, not an independent improvement estimate.
"""
import argparse
import ast
import itertools
import json
from pathlib import Path
import numpy as np


def metrics(y, p):
    cm = np.bincount(y*20+p, minlength=400).reshape(20,20)
    tp = cm.diagonal()
    return dict(oa=float(tp.sum()/len(y)), miou=float((tp/np.maximum(cm.sum(0)+cm.sum(1)-tp,1)).mean()),
                mean_recall=float((tp/np.maximum(cm.sum(1),1)).mean()),
                iou=(tp/np.maximum(cm.sum(0)+cm.sum(1)-tp,1)).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    root = args.folder
    names = next(ast.literal_eval(n.value) for n in ast.parse((Path(__file__).parent/'frozen_trajectory_probes.py').read_text()).body
                 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAMES' for t in n.targets))
    probabilities, y, indices = {}, None, None
    for path in sorted(root.glob('*/validation_predictions.npz')):
        with np.load(path) as data:
            if y is None:
                y, indices = data['labels'], data['indices']
            assert np.array_equal(y, data['labels']) and np.array_equal(indices,data['indices']), path
            z = data['logits'].astype(np.float64)
        assert z.shape == (len(y),20) and np.isfinite(z).all()
        z -= z.max(1,keepdims=True)
        p = np.exp(z)
        probabilities[path.parent.name] = p/p.sum(1,keepdims=True)
    assert len(probabilities) >= 2
    preds = {k:p.argmax(1) for k,p in probabilities.items()}
    scores = {k:metrics(y,p) for k,p in preds.items()}
    anchor = max(scores, key=lambda k:scores[k]['miou'])
    complement = {}
    for a,b in itertools.combinations(preds,2):
        ac, bc = preds[a]==y, preds[b]==y
        per_class = [dict(name=name, support=int((y==i).sum()),
            a_wrong_b_correct=int(((y==i)&~ac&bc).sum()),
            a_correct_b_wrong=int(((y==i)&ac&~bc).sum()),
            both_wrong=int(((y==i)&~ac&~bc).sum())) for i,name in enumerate(names)]
        complement[a+' vs '+b] = dict(a=a,b=b,disagreements=int((preds[a]!=preds[b]).sum()),
            a_wrong_b_correct=int((~ac&bc).sum()), a_correct_b_wrong=int((ac&~bc).sum()),
            both_wrong=int((~ac&~bc).sum()), oracle_correct_either=float((ac|bc).mean()), per_class=per_class)
    mixtures = []
    def add(weights):
        p = sum(probabilities[k]*w for k,w in weights.items())
        mixtures.append(dict(weights=weights, **metrics(y,p.argmax(1))))
    for k in probabilities:
        add({k:1.})
    for a,b in itertools.combinations(probabilities,2):
        for w in [.25,.5,.75]:
            add({a:w,b:1-w})
    if all(k in probabilities for k in ['baseline','capacity','grouped']):
        add({k:1/3 for k in ['baseline','capacity','grouped']})
    add({k:1/len(probabilities) for k in probabilities})
    mixtures.sort(key=lambda r:r['miou'],reverse=True)
    report=dict(validation_parcels=len(y), best_single=anchor, singles=scores,
                complementarity=complement, mixtures=mixtures,
                caveat='Live checkpoint snapshot; models have unequal training progress. Mixture selection and measurement share validation data already used for checkpoint selection. No test access. Oracle uses true labels and is not deployable.')
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Complementary validation errors','',report['caveat'],'',
           '| Model / probability mixture | OA % | mIoU % | Mean recall % |','|---|---:|---:|---:|']
    for name,s in list(scores.items())+[(str(r['weights']),r) for r in mixtures[:5]]:
        lines.append(f'| {name} | {100*s["oa"]:.3f} | {100*s["miou"]:.3f} | {100*s["mean_recall"]:.3f} |')
    lines += ['', '| Pair A / B | A wrong, B correct | A correct, B wrong | Both wrong |','|---|---:|---:|---:|']
    for name,r in complement.items():
        lines.append(f'| {name} | {r["a_wrong_b_correct"]} | {r["a_correct_b_wrong"]} | {r["both_wrong"]} |')
    lines += ['', f'Best mixture versus strongest single model ({anchor}):', '',
              '| Class | Validation parcels | Single IoU % | Mixture IoU % | Change, points |',
              '|---|---:|---:|---:|---:|']
    for i,name in enumerate(names):
        a,b = scores[anchor]['iou'][i], mixtures[0]['iou'][i]
        lines.append(f'| {name} | {int((y==i).sum())} | {100*a:.2f} | {100*b:.2f} | {100*(b-a):+.2f} |')
    lines += ['', 'Per-class complementary errors and every tested mixture: [report.json](report.json).']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()

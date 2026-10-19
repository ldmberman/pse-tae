"""Create a standalone report from completed multiclass probe predictions."""
import argparse
import base64
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def bootstrap(y, a, b, seed=42, repeats=1000):
    # Resample parcels within true classes, preserving rare-class test support.
    # Joint prediction cells preserve pairing while avoiding huge index arrays.
    rng = np.random.default_rng(seed)
    counts = np.bincount(y*400 + a*20 + b, minlength=8000).reshape(20,20,20)
    draws = np.zeros((repeats,20,20,20), dtype=np.int32)
    for k in range(20):
        n = int(counts[k].sum())
        if n:
            draws[:,k] = rng.multinomial(n, counts[k].ravel()/n, size=repeats).reshape(repeats,20,20)
    def scores(cm):
        tp = np.diagonal(cm, axis1=1, axis2=2)
        return np.mean(tp/np.maximum(cm.sum(1)+cm.sum(2)-tp,1), axis=1)
    delta = (scores(draws.sum(2)) - scores(draws.sum(3))) * 100
    return np.percentile(delta,[2.5,97.5]).tolist()


def main(root):
    result = json.loads((root/'results.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    models = result['models']; baseline = models[0]
    names = protocol['classes']
    winner = next(r for r in models if r['name']==result['winner'])
    pred = np.load(root/'test_predictions.npz')
    selections = {feature:max((r for r in models[1:] if r['features']==feature),key=lambda r:r['val']['miou'])
                  for feature in ['embedding','spectral','combined']}
    model_intervals = {r['name']:bootstrap(pred['labels'],pred['original_decoder'],pred[r['name']])
                       for r in models[1:]}
    intervals = {key:model_intervals[r['name']] for key,r in selections.items()}
    matched = []
    for family in ['logistic', 'xgb']:
        for weighting in ['natural', 'sqrt']:
            e = next(r for r in models if r['group']==f'embedding_{family}_{weighting}')
            es = next(r for r in models if r['group']==f'combined_{family}_{weighting}')
            matched.append(dict(family=family, weighting=weighting,
                                embedding_test_miou=100*e['test']['miou'],
                                combined_test_miou=100*es['test']['miou'],
                                test_miou_delta_pp=100*(es['test']['miou']-e['test']['miou']),
                                val_miou_delta_pp=100*(es['val']['miou']-e['val']['miou']),
                                test_oa_delta_pp=100*(es['test']['oa']-e['test']['oa'])))
    analysis = dict(selected_by_input={k:r['name'] for k,r in selections.items()},
                    matched_feature_comparisons=matched,
                    paired_stratified_bootstrap_miou_delta_pp_ci95=intervals,
                    model_miou_delta_pp_ci95=model_intervals,
                    bootstrap_note='1000 paired parcel bootstrap samples within true classes; conditional on fitted models and this fold, without multiplicity correction.')
    (root/'analysis.json').write_text(json.dumps(analysis,indent=2))
    rows=[]
    for r in models:
        mark=' ★' if r['name']==winner['name'] else ''
        ci=model_intervals.get(r['name'],[0,0])
        rows.append('<tr><td>'+html.escape(r['name']+mark)+'</td>'+''.join(
            f'<td>{100*r[s][k]:.2f}</td>' for s,k in [('val','miou'),('test','oa'),('test','miou'),('test','mean_recall')])+
            f'<td>{100*(r["test"]["miou"]-baseline["test"]["miou"]):+.2f}</td>'+
            f'<td>[{ci[0]:+.2f}, {ci[1]:+.2f}]</td></tr>')
    classes=[]
    for i,n in enumerate(names):
        classes.append('<tr><td>'+html.escape(n)+'</td>'+f'<td>{sum(baseline["test"]["confusion"][i])}</td>'+''.join(
            f'<td>{100*r["test"][k][i]:.2f}</td>' for r,k in [(baseline,'iou'),(winner,'iou'),(winner,'precision'),(winner,'recall')])+ '</tr>')
    matrix=np.array([np.array(r['test']['iou'])-baseline['test']['iou'] for r in selections.values()])*100
    fig,ax=plt.subplots(figsize=(14,4.5))
    limit=max(1,abs(matrix).max())
    im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(20),names,rotation=55,ha='right',fontsize=9)
    ax.set_yticks(range(3),['Embeddings','Spectral summaries','Combined'])
    for j in range(3):
        for i in range(20):
            ax.text(i,j,f'{matrix[j,i]:+.1f}',ha='center',va='center',fontsize=8,
                    color='white' if abs(matrix[j,i])>limit*.6 else 'black')
    fig.colorbar(im,ax=ax,label='IoU change (percentage points)')
    ax.set_title(f'Fold {protocol["fold"]} test: each input\'s classifier chosen by validation mIoU')
    fig.tight_layout(); fig.savefig(root/'class_iou_changes.png',dpi=180); fig.savefig(root/'class_iou_changes.pdf')
    image_source = 'data:image/png;base64,' + base64.b64encode((root/'class_iou_changes.png').read_bytes()).decode()
    cis=''.join(f'<li>{html.escape(k)}: {v[0]:+.2f} to {v[1]:+.2f} mIoU points</li>' for k,v in intervals.items())
    matched_rows=''.join(f'<tr><td>{r["family"]} / {r["weighting"]}</td>' + ''.join(
        f'<td>{r[k]:.2f}</td>' for k in ['embedding_test_miou','combined_test_miou',
                                       'test_miou_delta_pp','test_oa_delta_pp','val_miou_delta_pp'])+'</tr>' for r in matched)
    content=f'''<!doctype html><html><head><meta charset="utf-8"><title>Frozen multiclass probes</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:35px auto;padding:20px;line-height:1.5}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}</style></head><body>
<h1>Frozen multiclass probes — fold {protocol['fold']}</h1>
<p>Twenty classes compete for every parcel. ★ marks the classifier selected across the predefined grid using validation mIoU only. All scores below are percentages. Mean recall is unweighted class accuracy; OA is parcel-weighted accuracy.</p>
<table><tr><th>Classifier</th><th>Validation mIoU</th><th>Test OA</th><th>Test mIoU</th><th>Test mean recall</th><th>Δ test mIoU</th><th>95% CI for Δ</th></tr>{''.join(rows)}</table>
<h2>Matched comparisons: does adding spectral information help?</h2>
<p>Each row holds classifier family and weighting fixed. Both inputs receive the same hyperparameter search budget and select their settings on validation mIoU.</p>
<table><tr><th>Classifier / weighting</th><th>Embedding test mIoU</th><th>Combined test mIoU</th><th>Δ test mIoU</th><th>Δ test OA</th><th>Δ validation mIoU</th></tr>{matched_rows}</table>
<h2>Per-class changes</h2><img src="{image_source}">
<p>Each row chooses its classifier family, weight and hyperparameter by validation mIoU. This plot does not choose models separately for individual test classes.</p>
<h2>Validation-selected overall classifier: {html.escape(winner['name'])}</h2>
<table><tr><th>Class</th><th>Test support</th><th>Baseline IoU</th><th>Selected IoU</th><th>Selected precision</th><th>Selected recall</th></tr>{''.join(classes)}</table>
<h2>Uncertainty</h2><p>95% paired, class-stratified bootstrap intervals for each selected input\'s test mIoU difference from the decoder:</p><ul>{cis}</ul>
<p>{analysis['bootstrap_note']} Fold 1 was already examined during pairwise exploration, so this remains an exploratory comparison. Fold-to-fold confirmation is needed before claiming a robust improvement.</p>
<h2>Protocol and features</h2><p>Original seed-1 train/validation/test partition. Every classifier sees only training labels when fitting; standardization uses training features only. C ∈ {{0.01, 0.1}} for multinomial logistic regression. Boosted trees use depth 3 or 5, at most 300 rounds, and 40-round patience on validation mIoU. Natural frequencies and inverse-square-root training class weights (normalized to mean one) are separate variants. All models use CPU; original checkpoints are unchanged.</p>
<p>The decoder reference uses the same cached, deterministic pixel samples as these probes, with inference jitter disabled. Its scores can therefore differ slightly from the previously reported baseline evaluation. This is a finite tuning-budget pilot: a tree reaching the 300-round limit has not necessarily exhausted its potential.</p>
<p>Embedding: 128-dimensional output of the frozen temporal encoder. Spectral: 832 values comprising all 24 dates\' parcel band means and standard deviations, four geometric attributes, and red-edge/SWIR index values, daily changes and seasonal summaries. Combined: 960 values. Spectral statistics use all parcel pixels; the encoder uses a deterministic sample of 64. This sampling difference may contribute to any gain.</p>
<p>For parcel band means μ: NDVI=(μB8−μB4)/(μB8+μB4) summarizes vegetation contrast; NDRE5/6/7=(μB8A−μBk)/(μB8A+μBk), k=5,6,7, summarize near-infrared/red-edge contrast associated with canopy chlorophyll and structure; NDMI=(μB8A−μB11)/(μB8A+μB11) summarizes vegetation moisture-related contrast; ND_SWIR=(μB11−μB12)/(μB11+μB12) describes SWIR spectral contrast influenced by vegetation, moisture and soil. These are proxies, not direct physical measurements. Ratios are computed before feature standardization, with zero denominators mapped to zero. The standard Sentinel-2 band order is assumed because the archive has no explicit band-name metadata.</p>
<p>Temporal features retain all dated indices, adjacent changes divided by elapsed days, time-weighted mean, min/max/amplitude, peak offset from first date divided by 365, extreme slopes, and Jan–Mar/Apr–Jun/Jul–Aug/Sep–Oct means. See <a href="feature_definitions.json">feature_definitions.json</a> for the complete feature names.</p>
</body></html>'''
    (root/'report.html').write_text(content)
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    main(p.parse_args().directory)

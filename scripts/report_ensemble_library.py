"""Render frozen validation selection and its held-out comparison."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ensemble_library/Fold_1'
s=json.loads((OUT/'selection.json').read_text());r=json.loads((OUT/'report.json').read_text());keys=['miou','oa','mean_recall']
labels={'six_model_reference':'Previous six-model ensemble','eight_checkpoint_control':'Six + baseline checkpoints 27 / 78','validation_selected':f"Validation-selected ensemble ({6+len(s['selected'])} members)"}
rows=list(labels);maximum={k:max(r['metrics'][n][k] for n in rows) for k in keys}
lines=['Ensemble library comparison','','Fold 1; 38,283 validation parcels and 38,283 disjoint test parcels. Existing checkpoints only. All values below are percentages or percentage-point differences.','','| Fixed comparison | Test mIoU | Test OA | Test average recall |','|---|---:|---:|---:|']
for n,label in labels.items():
 m=r['metrics'][n];cells=[f"{100*m[k]:.2f}" for k in keys];cells=[f'**{v}**' if m[k]==maximum[k] else v for k,v in zip(keys,cells)];lines.append('| '+label+' | '+' | '.join(cells)+' |')
lines+=['',f"Added members: {', '.join(s['selected'])}. Total weight of added group: {r['alpha']:.6f}; equal weights within each group.", 'Selection was frozen before this test evaluation. The checkpoint control was specified in advance. No test result was used to pick members or weights.','','| Comparison with six-model reference | Δ test mIoU | Paired 95% interval |','|---|---:|---:|']
for n in ['eight_checkpoint_control','validation_selected']:
 d=r['paired_miou_differences'][n];lines.append(f"| {labels[n]} | {100*d['delta']:+.2f} | [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}] |")
lines+=['','1,000 paired parcel-bootstrap draws. Conditional on fitted models; not fold, seed, spatial, or model-selection uncertainty. Fold 1 and validation have been reused throughout this exploratory work.','','Each candidate added separately to the unchanged six-model reference','','| Candidate | Standalone validation mIoU | Seven-model validation mIoU | Δ validation mIoU |','|---|---:|---:|---:|']
for x in s['single_addition_ranking']:lines.append(f"| {x['name']} | {100*x['standalone']['miou']:.2f} | {100*x['addition']['miou']:.2f} | {100*x['delta']:+.2f} |")
lines+=['','Greedy selection: at most four additions, minimum +0.05 validation mIoU points per addition.','','| Step | Candidate | Resulting validation mIoU | Marginal gain | Accepted |','|---:|---|---:|---:|---|']
for x in s['greedy_trace']:lines.append(f"| {x['step']} | {x['candidate']} | {100*x['validation_miou']:.2f} | {100*x['delta']:+.2f} | {x['accepted']} |")
lines+=['','Group-weight comparison: the original six retain equal internal weights; so do the added members.','','| Total added-group weight | Validation mIoU | Validation OA | Validation average recall |','|---:|---:|---:|---:|']
for x in s['weight_trials']:lines.append(f"| {x['alpha']:.4f} | "+' | '.join(f"{100*x['metrics'][k]:.2f}" for k in keys)+' |')
classes=json.loads((ROOT/'results/multiclass_probes/Fold_1/protocol.json').read_text())['classes'];delta=r['per_class_iou_delta']
lines+=['',f"Selected ensemble: {sum(d>0 for d in delta)} class IoUs improve, {sum(d<0 for d in delta)} worsen, {sum(d==0 for d in delta)} tie.",'','| Class | Six-model test IoU | Selected test IoU | Change |','|---|---:|---:|---:|']
for i,n in enumerate(classes):lines.append(f"| {n} | {100*r['metrics']['six_model_reference']['iou'][i]:.2f} | {100*r['metrics']['validation_selected']['iou'][i]:.2f} | {100*delta[i]:+.2f} |")
lines+=['','[Predeclared selection protocol](protocol.json) · [Frozen selection and validation input hashes](selection.json) · [Test metrics and input hashes](report.json)','']
(OUT/'report.md').write_text('\n'.join(lines))
print('Wrote',OUT/'report.md')

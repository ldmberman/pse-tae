"""Verify all ten saved test outputs, then publish their five-fold measurement tables."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from learning.splits import partitions
from scripts.report_saved import markdown,svg


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def measured(y,p):
    cm=np.bincount(y*20+p,minlength=400).reshape(20,20);tp=cm.diagonal()
    assert np.all(cm.sum(1)>0),'All 20 classes must be represented in this test fold'
    return dict(miou=100*float(np.mean(tp/(cm.sum(0)+cm.sum(1)-tp))),
                oa=100*float(tp.sum()/cm.sum()),mean_recall=100*float(np.mean(tp/cm.sum(1))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=Path('outputs/xgboost_60_20_20'))
    p.add_argument('--features',type=Path,default=Path('results/xgboost'))
    p.add_argument('--readme',type=Path,default=Path('README.md'));a=p.parse_args()
    y=np.load(a.features/'targets.npy');result={}
    for name in ['mean_std','mean_std_kurtosis']:
        rows=[]
        for fold in range(1,6):
            folder=a.runs/name/f'Fold_{fold}';r=json.loads((folder/'metrics.json').read_text())
            expected=partitions(len(y),fold);saved=np.load(folder/'splits.npz')
            for key in expected:np.testing.assert_array_equal(expected[key],saved[key])
            for split,filename in [('test','predictions.npz'),('val','validation_predictions.npz')]:
                d=np.load(folder/filename);ids=expected[split]
                np.testing.assert_array_equal(d['indices'],ids);np.testing.assert_array_equal(d['labels'],y[ids])
                values=measured(y[ids],d['probabilities'].argmax(1))
                if split=='test':
                    for k,v in values.items():assert np.isclose(v,r['test'][k],rtol=0,atol=1e-9)
                else:assert np.isclose(values['miou'],r['validation_miou'],rtol=0,atol=1e-9)
            history=json.loads((folder/'history.json').read_text())['validation']['miou']
            assert r['best_round']==int(np.argmax(history))+1
            assert r['protocol']['split']=='60/20/20' and r['protocol']['fold']==fold
            rows.append(dict(r,model_sha256=digest(folder/'model.ubj'),
                             split_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in expected.items()}))
        overall=dict(folds=rows,summary={k:dict(mean=float(np.mean([r['test'][k] for r in rows])),sd=float(np.std([r['test'][k] for r in rows],ddof=1))) for k in ['miou','oa','mean_recall']})
        (a.runs/name/'overall.json').write_text(json.dumps(overall,indent=2)+'\n');result[name]=overall
    result['provenance']=dict(feature_cache_sha256={f:digest(a.features/f) for f in ['moments.npy','targets.npy','pids.npy']},
                              execution='CPU; exact software, architecture and thread counts are recorded per fold when available',
                              notes='Exact original 60/20/20 partitions; validation-selected boosting round. Legacy <2-pixel parcels retain the original all-zero moment vector. Across-fold SD is descriptive; training sets overlap.')
    target=Path('records/xgboost_60_20_20.json');target.write_text(json.dumps(result,indent=2)+'\n')
    ledger=Path('records/table_data.json');panels=json.loads(ledger.read_text())
    panels=[panel for panel in panels if not panel['title'].startswith('Historical XGBoost')]
    for panel in panels[:2]:
        for _,rows in panel['groups']:
            rows[:]=[r for r in rows if r['label'] not in ['XGBoost on band mean / std','XGBoost on band mean / std + kurtosis']
                     and not r['label'].startswith('XGBoost 60/20/20:')]
    for name,label in [('mean_std','band mean / std'),('mean_std_kurtosis','band mean / std + kurtosis')]:
        summary=result[name]['summary'];panels[1]['groups'][1][1].append(dict(label='XGBoost 60/20/20: '+label,
             values=[summary[k]['mean'] for k in ['miou','oa','mean_recall']],sd=[summary[k]['sd'] for k in ['miou','oa','mean_recall']],source=str(target)))
    ledger.write_text(json.dumps(panels,indent=2)+'\n')
    outputs=Path('outputs');outputs.mkdir(parents=True,exist_ok=True)
    (outputs/'recorded_results.md').write_text(markdown(panels).rstrip()+'\n');(outputs/'recorded-results.svg').write_text(svg(panels)[0]+'\n')
    baseline=panels[1]['groups'][0][1][0]
    table=['Five folds: held-out test, mean ± sample standard deviation.','','| Model | Test mIoU | Test OA | Test average recall |','|---|---:|---:|---:|']
    table_rows=[('PSE+TAE reproduction',baseline['values'],baseline['sd'])]
    for name,label in [('mean_std','XGBoost: band mean/std'),('mean_std_kurtosis','XGBoost: band mean/std + kurtosis')]:
        summary=result[name]['summary'];table_rows.append((label,[summary[k]['mean'] for k in ['miou','oa','mean_recall']],[summary[k]['sd'] for k in ['miou','oa','mean_recall']]))
    maxima=[max(row[1][i] for row in table_rows) for i in range(3)]
    for label,means,sds in table_rows:
        cells=[('**' if v==maxima[i] else '')+f'{v:.2f}'+('**' if v==maxima[i] else '')+f' ± {sds[i]:.2f}' for i,v in enumerate(means)]
        table.append('| '+label+' | '+' | '.join(cells)+' |')
    table+=['| Paper\'s random forest | 32.5 | 91.6 | - |', '', '[Per-fold scores, selected rounds, partitions and checkpoint hashes](records/xgboost_60_20_20.json).']
    s=a.readme.read_text();start='<!-- XGB_60_20_20_RESULTS -->';end='<!-- END_XGB_60_20_20_RESULTS -->'
    if start not in s:raise ValueError('Missing README insertion marker')
    begin=s.index(start)+len(start)
    finish=s.index(end,begin) if end in s else begin
    s=s[:begin]+'\n\n'+'\n'.join(table)+'\n\n'+(s[finish:] if end in s else end+s[finish:])
    a.readme.write_text(s.rstrip()+'\n')
    print(json.dumps({k:v['summary'] for k,v in result.items() if k!='provenance'},indent=2))
if __name__=='__main__':main()

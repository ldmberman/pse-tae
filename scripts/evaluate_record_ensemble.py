"""Test the immutable, validation-selected ensemble; no tuning on test labels."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.train_tgt_scratch import PatchInputs,build
from scripts.tgt_correction_probes import batches,predict,to_device
from scripts.frozen_trajectory_probes import metrics,write_json,NAMES
from scripts.train_pixel_temporal import runtime


def save_predictions(path, indices, labels, logits):
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,indices=indices,labels=labels,logits=logits)
    path.with_suffix('.tmp').replace(path)


def softmax(z):
    z=z.astype(np.float64);z-=z.max(1,keepdims=True)
    p=np.exp(z)
    return p/p.sum(1,keepdims=True)


def run(args):
    root=Path(args.snapshot)
    output=root/'test_evaluation';output.mkdir(exist_ok=True)
    status_path=output/('status_'+args.only_model+'.json' if args.only_model else 'status.json')
    with (output/('run_'+args.only_model+'.lock' if args.only_model else 'run.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime(1);torch.set_num_threads(args.threads)
        manifest=json.loads((root/'ensemble.json').read_text())
        variants=list(manifest['models'])
        assert variants==['baseline','capacity','grouped']
        assert all(abs(r['weight']-1/3)<1e-12 for r in manifest['models'].values())
        options={k:v for k,v in vars(args).items() if k!='only_model'}
        protocol=dict(options,selection='Fixed epoch38/67/38 checkpoints and equal probability average selected on validation before test access.',
            checkpoint_hashes={v:r['checkpoint_sha256'] for v,r in manifest['models'].items()},
            combination=manifest['combination'],bootstrap_seed=20261018,bootstrap_replicates=1000,
            caveat='Exploratory previously inspected fold1; no weights or checkpoints selected using this test evaluation. CPU inference; parcel bootstrap conditional on fitted models, not seed/fold/spatial uncertainty.')
        if (output/'protocol.json').exists():
            assert json.loads((output/'protocol.json').read_text())==protocol
        else:
            write_json(output/'protocol.json',protocol)
        write_json(status_path,dict(stage='initializing',updated_unix=time.time()))
        try:
            if args.only_model is None:
                jobs=[]
                write_json(status_path,dict(stage='parallel_inference',updated_unix=time.time()))
                for v in variants:
                    if (output/(v+'_predictions.npz')).exists() and (output/(v+'_provenance.json')).exists():
                        continue
                    with (output/(v+'.log')).open('a') as log:
                        job=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],'--only-model',v],stdout=log,stderr=subprocess.STDOUT)
                    jobs.append((v,job))
                failed=[]
                for v,job in jobs:
                    code=job.wait()
                    if code:failed.append((v,code))
                assert not failed,failed
            ds=PatchInputs(args);test=ds.splits['test'];y=ds.y[test]
            probabilities={};scores={};preds={};checks={}
            for v in variants:
                if args.only_model and v!=args.only_model:
                    continue
                folder=root/v
                digest=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()
                assert digest==manifest['models'][v]['checkpoint_sha256']
                ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
                assert ck['epoch']==manifest['models'][v]['epoch']
                source=ck['protocol']
                assert source['pid_sha256']==hashlib.sha256(','.join(ds.pids).encode()).hexdigest()
                assert source['split_hashes']=={k:hashlib.sha256(ix.tobytes()).hexdigest() for k,ix in ds.splits.items()}
                for name,actual in [('spectral_mean',ds.mean),('spectral_std',ds.std),('days',ds.days),
                                    ('geometry_mean',ds.geom_mean),('geometry_std',ds.geom_std)]:
                    np.testing.assert_allclose(source[name],actual,rtol=0,atol=0)
                target=output/(v+'_predictions.npz')
                meta_path=output/(v+'_provenance.json')
                if target.exists() and meta_path.exists():
                    provenance=json.loads(meta_path.read_text());assert provenance['checkpoint_sha256']==digest
                    with np.load(target) as f:
                        assert np.array_equal(f['indices'],test) and np.array_equal(f['labels'],y)
                        logits=f['logits']
                    checks[v]=provenance['validation_check']
                else:
                    model=build(ds,v).to(args.device);model.load_state_dict(ck['state_dict']);model.eval()
                    probe=ds.splits['val'][:64]
                    reference,labels,ids=predict(model,batches(ds,probe.tolist(),args),args.device)
                    with np.load(folder/'validation_predictions.npz') as f:
                        assert np.array_equal(f['indices'][:64],ids) and np.array_equal(f['labels'][:64],labels)
                        saved=f['logits'][:64]
                    checks[v]=dict(parcels=64,max_abs_logit_difference=float(abs(reference-saved).max()),
                        changed_predictions=int((reference.argmax(1)!=saved.argmax(1)).sum()))
                    np.testing.assert_allclose(reference,saved,rtol=1e-4,atol=1e-4)
                    assert checks[v]['changed_predictions']==0
                    logits=np.empty((len(test),20),np.float32);offset=0;tick=time.time()
                    def infer_batch(item):
                        batch,labels,ids=item
                        with torch.inference_mode():
                            out=model(**to_device(batch,args.device))
                            assert torch.isfinite(out).all()
                            return out.cpu().numpy(),labels.numpy(),ids.numpy()
                    def consume(result):
                        nonlocal offset
                        out,labels,ids=result;n=len(labels)
                        assert np.array_equal(ids,test[offset:offset+n]) and np.array_equal(labels,y[offset:offset+n])
                        logits[offset:offset+n]=out;offset+=n
                        if offset%(100*args.batch_size)==0 or offset==len(test):
                            state=dict(stage='inference',model=v,parcels=offset,total=len(test),seconds=time.time()-tick,updated_unix=time.time())
                            write_json(status_path,state);print(json.dumps(state),flush=True)
                    # Start loader workers before inference threads. Each thread
                    # reads the same eval-mode weights; no model state is mutated.
                    iterator=iter(batches(ds,test.tolist(),args))
                    pending=deque()
                    with ThreadPoolExecutor(max_workers=args.inference_workers,initializer=lambda:torch.set_num_threads(args.threads)) as pool:
                        for item in iterator:
                            pending.append(pool.submit(infer_batch,item))
                            if len(pending)>=2*args.inference_workers:
                                consume(pending.popleft().result())
                        while pending:
                            consume(pending.popleft().result())
                    assert offset==len(test)
                    save_predictions(target,test,y,logits)
                    write_json(meta_path,dict(checkpoint_sha256=digest,epoch=ck['epoch'],device=args.device,validation_check=checks[v],seconds=time.time()-tick))
                    del model
                probabilities[v]=softmax(logits);preds[v]=logits.argmax(1);scores[v]=metrics(y,preds[v])
                print(json.dumps(dict(model=v,test={k:scores[v][k] for k in ['oa','miou','mean_recall']})),flush=True)
            if args.only_model:
                write_json(status_path,dict(stage='complete',model=args.only_model,updated_unix=time.time()))
                return
            # The two-model comparator was already defined and evaluated on
            # validation. No new mixture weights are searched here.
            mixtures={'baseline_capacity':(probabilities['baseline']+probabilities['capacity'])/2,
                'ensemble':sum(probabilities[v]*manifest['models'][v]['weight'] for v in variants)}
            for v,p in mixtures.items():
                preds[v]=p.argmax(1);scores[v]=metrics(y,preds[v])
                save_predictions(output/(v+'_predictions.npz'),test,y,np.log(p.clip(1e-300)))
            with np.load(Path(args.cache)/f'fold_{args.fold}_features.npz') as f:
                preds['historical_native_baseline']=f['logits'][test].argmax(1)
            scores['historical_native_baseline']=metrics(y,preds['historical_native_baseline'])
            pairs=[('ensemble',v) for v in ['baseline','capacity','grouped','baseline_capacity','historical_native_baseline']]
            draws={a+'-minus-'+b:[] for a,b in pairs};rng=np.random.RandomState(20261018)
            write_json(output/'status.json',dict(stage='bootstrap',updated_unix=time.time()))
            for _ in range(1000):
                ix=rng.randint(len(y),size=len(y));m={v:metrics(y[ix],p[ix])['miou'] for v,p in preds.items()}
                for a,b in pairs:draws[a+'-minus-'+b].append(m[a]-m[b])
            report=dict(test_parcels=len(test),validation_record=manifest['validation'],protocol=protocol,metrics=scores,
                validation_checks=checks,paired_miou_differences={a+'-minus-'+b:dict(delta=scores[a]['miou']-scores[b]['miou'],
                    ci95=np.quantile(draws[a+'-minus-'+b],[.025,.975]).tolist()) for a,b in pairs})
            write_json(output/'report.json',report)
            lines=['# Fixed record ensemble: fold 1 test','',protocol['caveat'],'',
                '| Model | Test OA % | Test mIoU % | Mean class recall % |','|---|---:|---:|---:|']
            for v,s in scores.items():lines.append(f'| {v} | {100*s["oa"]:.3f} | {100*s["miou"]:.3f} | {100*s["mean_recall"]:.3f} |')
            lines+=['','Historical native baseline uses the earlier fixed 64-pixel cache; other models use the same full patches.','',
                '| Comparison | mIoU difference, points | Paired 95% bootstrap interval |','|---|---:|---|']
            for name,r in report['paired_miou_differences'].items():lines.append(f'| {name} | {100*r["delta"]:+.3f} | [{100*r["ci95"][0]:+.3f}, {100*r["ci95"][1]:+.3f}] |')
            lines+=['','| Class | Test parcels | Baseline IoU % | Capacity IoU % | Grouped IoU % | Ensemble IoU % |','|---|---:|---:|---:|---:|---:|']
            for i,name in enumerate(NAMES):lines.append(f'| {name} | {int((y==i).sum())} | '+' | '.join(f'{100*scores[v]["iou"][i]:.2f}' for v in variants+['ensemble'])+' |')
            (output/'report.md').write_text('\n'.join(lines)+'\n')
            write_json(output/'status.json',dict(stage='complete',updated_unix=time.time(),metrics={k:scores['ensemble'][k] for k in ['oa','miou','mean_recall']}))
            print('\n'.join(lines),flush=True)
        except Exception as exc:
            write_json(output/'status.json',dict(stage='failed',error=repr(exc),updated_unix=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',default='results/complementarity/snapshot_20261018_1605')
    p.add_argument('--cache',default='results/pair_probes')
    p.add_argument('--native',default='data/S2-2017-T31TFM-PixelSet')
    p.add_argument('--patch',default='data/s2-2017-IGARSS-NNI-NPY')
    p.add_argument('--fold',type=int,default=1)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--inference-workers',type=int,default=4)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--only-model',choices=['baseline','capacity','grouped'])
    run(p.parse_args())

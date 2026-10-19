"""Evaluate one saved checkpoint on its own fold; no fitting or selection.

Native models use the fixed 64-pixel draws from our later cached comparisons.
A --limit run is explicitly marked partial, never a full test measurement.
"""
import argparse, hashlib, inspect, json, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from scripts.frozen_pair_probes import make_dataset, partitions, NAMES
from scripts.frozen_trajectory_probes import metrics


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def move(x, device):
    if isinstance(x, dict): return {k:move(v,device) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return type(x)(move(v,device) for v in x)
    return x.to(device)


def check_fold(ck, path, fold):
    declared=ck.get('protocol',{}).get('fold')
    path_folds=[int(m.group(1)) for p in Path(path).parts if (m := re.fullmatch(r'fold[_-](\d+)',p,re.I))]
    if declared is not None and declared != fold: raise ValueError('Checkpoint belongs to another fold')
    if path_folds and any(f!=fold for f in path_folds): raise ValueError('Checkpoint path belongs to another fold')
    if declared is None and not path_folds: raise ValueError('No checkpoint fold provenance; use a Fold_N or fold-N directory')


def model_arguments(a):
    if a.experiment:
        recipes=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments.json').read_text())
        recipe=recipes[a.experiment]
        if recipe['kind']!='native' or recipe['family']!=a.family:
            raise ValueError('Experiment configuration does not match the evaluation family')
        return dict(recipe['model_kwargs'])
    return json.loads(Path(a.model_config).read_text()) if a.model_config else {}


def run(a):
    torch.set_num_threads(a.threads);torch.manual_seed(1)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    check_fold(ck,a.checkpoint,a.fold)
    model_kwargs=model_arguments(a)
    if a.family in ['native','ltae','rope','distilled']:
        from models.stclassifier import PseTae,PseLTae
        ds=make_dataset(a);y=np.asarray(ds.target);pids=np.asarray(ds.pid)
        constructor=PseLTae if a.family=='ltae' else PseTae
        allowed=inspect.signature(constructor).parameters
        unknown=set(model_kwargs)-set(allowed)
        if unknown:raise ValueError(f'Unknown model arguments: {unknown}')
        model_kwargs['positions']=ds.date_positions
        if a.family=='rope':model_kwargs['pos_encoding']='rope'
        model=constructor(**model_kwargs)
        native=True
    elif a.family in ['patch','large','dropout','cnn_large','cnn_correction','group_correction']:
        from scripts.train_tgt_scratch import PatchInputs
        from scripts.cnn_correction_probes import GroupInputs
        if a.family.endswith('correction'):
            ds=GroupInputs(a,a.variant,norm=ck['normalization'])
            from models.cnn_correction import CNNCorrection
            from models.tgt_correction import GroupedCorrection
            model=(CNNCorrection if a.family=='cnn_correction' else GroupedCorrection)(ds.days,ds.mean,ds.std,a.variant)
        else:
            ds=PatchInputs(a)
            from models.tgt_scratch import GroupedScratch
            from models.capacity_large import CapacityLarge
            from models.date_dropout import DateDropoutCapacity
            from models.cnn_ta_td_large import CNNTATDLarge
            constructors={'large':CapacityLarge,'dropout':DateDropoutCapacity,'cnn_large':CNNTATDLarge}
            model=GroupedScratch(ds.days,ds.mean,ds.std,a.variant) if a.family=='patch' else constructors[a.family](ds.days,ds.mean,ds.std)
        y=ds.y;pids=ds.pids;native=False
        pr=ck.get('protocol',{})
        if 'pid_sha256' in pr:assert hashlib.sha256(','.join(pids).encode()).hexdigest()==pr['pid_sha256']
        for key in ['split_hashes','split_sha256']:
            if key in pr:assert pr[key]=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in ds.splits.items()}
    elif a.family in ['small_correction','small_probe']:
        from scripts.frozen_trajectory_probes import Inputs,Probe
        from scripts.frozen_correction_probes import CorrectionInputs,Correction
        ds=(CorrectionInputs if a.family=='small_correction' else Inputs)(a,a.variant,norm=ck['normalization'])
        model=(Correction if a.family=='small_correction' else Probe)(a.variant,ds.data['days'])
        y=ds.y;pids=ds.data['pids'];native=False
    else:
        from scripts.train_pixel_temporal import Parcels
        from models.pixel_temporal import PixelTemporalClassifier
        a.features=a.cache;a.pixels=a.trajectories
        ds=Parcels(a,training=False);y=ds.y;pids=ds.pids
        model=PixelTemporalClassifier(ds.days,a.variant=='temporal_spatial_nopos');native=False
    state=dict(ck.get('state_dict',ck.get('model')))
    if a.legacy_unscaled_kurtosis:
        key='spatial_encoder.kurtosis_gap'
        if key not in state:
            assert a.family=='native' and 'kurtosis' in model_kwargs.get('pooling','')
            state[key]=torch.ones_like(model.state_dict()[key])
    model.load_state_dict(state,strict=True);model.to(a.device).eval()
    split=partitions(len(y),a.fold);ids=split[a.split]
    if a.limit:ids=ids[:a.limit]
    output=[]
    with torch.inference_mode():
        if a.family in ['small_correction','small_probe']:
            for i in range(0,len(ids),a.batch_size):
                x,_=ds.batch(ids[i:i+a.batch_size],a.device);output.append(model(**x).cpu().numpy())
        else:
            loader=DataLoader(Subset(ds,ids.tolist()),batch_size=a.batch_size,num_workers=a.workers,shuffle=False)
            offset=0
            for batch in loader:
                if a.family=='pixel_temporal':
                    x,mask,geometry,labels,ix=batch
                    z=model(x.to(a.device),mask.to(a.device),geometry.to(a.device))
                else:
                    x,labels,ix=batch;x=move(x,a.device)
                    z=model(x) if native else model(**x)
                assert np.array_equal(ix.numpy(),ids[offset:offset+len(ix)])
                assert np.array_equal(labels.numpy(),y[ix.numpy()])
                output.append(z.cpu().numpy());offset+=len(ix)
                if offset%(a.batch_size*100)==0:print(f'{offset}/{len(ids)} parcels',flush=True)
    z=np.concatenate(output);assert np.isfinite(z).all()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    meta=dict(fold=a.fold,split=a.split,family=a.family,variant=a.variant,checkpoint_sha256=digest(a.checkpoint),epoch=ck.get('epoch'),partial=bool(a.limit),pixel_sampling='fixed seed 20261013 + parcel ID for native models',n=len(ids))
    np.savez_compressed(out/'predictions.npz',indices=ids,pids=pids[ids],labels=y[ids],logits=z,fold=a.fold,split=a.split,partial=bool(a.limit))
    report=dict(**meta,classes=NAMES,metrics=metrics(y[ids],z.argmax(1)))
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:100*report['metrics'][k] for k in ['miou','oa','mean_recall']},indent=2))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--family',required=True,choices=['native','ltae','rope','distilled','patch','large','dropout','cnn_large','cnn_correction','group_correction','small_correction','small_probe','pixel_temporal'])
    p.add_argument('--variant',default='baseline')
    config=p.add_mutually_exclusive_group()
    config.add_argument('--experiment',help='Native experiment name in configs/experiments.json')
    config.add_argument('--model-config',help='Optional custom model-argument JSON file')
    p.add_argument('--legacy-unscaled-kurtosis',action='store_true',help='Explicit compatibility for early checkpoints predating the scale buffer; uses scale=1')
    p.add_argument('--dataset',default='data/S2-2017-T31TFM-PixelSet');p.add_argument('--native',default='data/S2-2017-T31TFM-PixelSet');p.add_argument('--patch',default='data/s2-2017-IGARSS-NNI-NPY')
    p.add_argument('--cache',default='results/pair_probes');p.add_argument('--trajectories',default='results/frozen_trajectories/native_cache');p.add_argument('--patch-trajectories',default='results/frozen_trajectories/patch_cache')
    p.add_argument('--fold',type=int,choices=range(1,6),required=True);p.add_argument('--split',choices=['train','val','test'],default='test')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=4);p.add_argument('--workers',type=int,default=0);p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=1);p.add_argument('--limit',type=int,default=0);p.add_argument('--output',required=True)
    return p
if __name__=='__main__':run(parser().parse_args())

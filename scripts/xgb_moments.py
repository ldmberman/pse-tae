"""XGBoost on the paper's per-date band moments, same 5 folds as train.py."""

import argparse
from pathlib import Path
import json
import os
import sys
import time
import gc
import platform
from concurrent.futures import ProcessPoolExecutor

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.splits import partitions

ROOT = "data/S2-2017-T31TFM-PixelSet"
OUT = "results/xgboost"
SUB_CLASSES = [1, 3, 4, 5, 6, 8, 9, 12, 13, 14, 16, 18, 19, 23, 28, 31, 33, 34, 36, 39]
N_DATES = 24
N_BANDS = 10
N_MOMENTS = 5
WORKERS = 2
FEATURES = None
CHUNK = 800


def miou(y_true, y_pred, n_classes=20):
    total = 0.0
    seen = 0
    for c in range(n_classes):
        truth = y_true == c
        pred = y_pred == c
        union = np.logical_or(truth, pred).sum()
        if union == 0:
            continue
        total += np.logical_and(truth, pred).sum() / union
        seen += 1
    return float(total / seen)


def parcel_moments(path):
    values = np.load(path, mmap_mode="r")
    cube = np.asarray(values, dtype=np.float64)
    if cube.ndim != 3 or cube.shape[0] != N_DATES or cube.shape[1] != N_BANDS or cube.shape[2] < 2:
        return np.zeros((N_DATES, N_BANDS, N_MOMENTS), np.float32)
    mean = cube.mean(axis=-1)
    centered = cube - mean[..., None]
    second = np.mean(centered * centered, axis=-1)
    std = np.sqrt(second)
    valid = std > 1e-6
    scale = np.where(valid, std, 1.0)
    normal = centered / scale[..., None]
    skew = np.mean(normal ** 3, axis=-1)
    kurtosis = np.mean(normal ** 4, axis=-1) - 3.0
    fifth = np.mean(normal ** 5, axis=-1)
    skew = np.where(valid, skew, 0.0)
    kurtosis = np.where(valid, kurtosis, 0.0)
    fifth = np.where(valid, fifth, 0.0)
    std = np.where(valid, std, 0.0)
    return np.stack([mean, std, skew, kurtosis, fifth], axis=-1).astype(np.float32)


def compute_chunk(paths):
    return np.stack([parcel_moments(path) for path in paths], axis=0)


def dataset_index():
    if FEATURES and not (Path(ROOT)/'DATA').is_dir():
        # A copied feature cache includes the exact original parcel order.
        pids=np.load(Path(FEATURES)/'pids.npy')
        targets=np.load(Path(FEATURES)/'targets.npy')
        if len(pids)!=len(targets) or len(set(pids))!=len(pids):raise ValueError('Invalid cached parcel index')
        return pids.tolist(),targets
    labels = json.load(open(os.path.join(ROOT, "META", "labels.json")))["label_44class"]
    names = [int(name[:-4]) for name in os.listdir(os.path.join(ROOT, "DATA")) if name.endswith(".npy")]
    order = [str(pid) for pid in np.sort(np.array(names))]
    kept = []
    targets = []
    convert = {code: index for index, code in enumerate(SUB_CLASSES)}
    for pid in order:
        code = labels[pid]
        if code in convert:
            kept.append(pid)
            targets.append(convert[code])
    return kept, np.array(targets, dtype=np.int32)


def folds(n_samples):
    return [partitions(n_samples, fold) for fold in range(1, 6)]


def flatten(moments, which):
    chosen = moments[:, :, :, which]
    return np.ascontiguousarray(chosen.reshape(chosen.shape[0], -1))


def train_set(name, features, targets, fold_list):
    print("FEATURE_SET {} shape {}".format(name, features.shape), flush=True)
    records=[]
    params=dict(max_depth=6, eta=.1, subsample=.8, colsample_bytree=.8,
                objective='multi:softprob', num_class=20, tree_method='hist',
                device='cpu', nthread=WORKERS, seed=1, disable_default_eval_metric=1)
    for fold, split in enumerate(fold_list, start=1):
        if FOLD and fold != FOLD: continue
        train_ids, val_ids, test_ids = (split[s] for s in ['train','val','test'])
        dest = Path(OUT) / name / f'Fold_{fold}'
        dest.mkdir(parents=True, exist_ok=True)
        protocol=dict(split='60/20/20', selection='maximum validation mIoU; earliest tie',
                      rounds=ROUNDS, params=params, fold=fold, feature_set=name,
                      counts={s:len(ids) for s,ids in split.items()})
        result_path=dest/'metrics.json'
        if ACTION=='train' and result_path.exists():
            saved=json.loads(result_path.read_text())
            previous=json.loads(json.dumps(saved['protocol']));requested=json.loads(json.dumps(protocol))
            # Worker count is an execution setting. Preserve the completed run's
            # recorded count while permitting resume on a different CPU budget.
            previous['params'].pop('nthread',None);requested['params'].pop('nthread',None)
            if previous != requested: raise ValueError('Existing run has a different protocol')
            records.append(saved);print('SKIP completed',name,fold,flush=True);continue
        started=time.time()
        if ACTION == 'train':
            dtrain=xgb.QuantileDMatrix(features[train_ids],label=targets[train_ids],nthread=WORKERS,max_bin=256)
            dval=xgb.QuantileDMatrix(features[val_ids],label=targets[val_ids],ref=dtrain,nthread=WORKERS,max_bin=256)
            history={}
            model=xgb.train(params,dtrain,num_boost_round=ROUNDS,evals=[(dval,'validation')],
                            custom_metric=lambda pred,d:('miou',miou(d.get_label().astype(int),pred.argmax(1))),
                            evals_result=history,verbose_eval=10)
            best=int(np.argmax(history['validation']['miou']))+1
            model=model[:best]
            model.save_model(dest/'model.ubj')
            (dest/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            np.savez_compressed(dest/'splits.npz',**split)
            del dtrain,dval;gc.collect()
        else:
            saved=json.loads(result_path.read_text())
            expected=np.load(dest/'splits.npz')
            if any(not np.array_equal(expected[k],v) for k,v in split.items()):raise ValueError('Saved partitions differ from the requested fold')
            if saved['protocol']['split']!='60/20/20' or saved['protocol']['feature_set']!=name:raise ValueError('Incompatible saved protocol')
            protocol=saved['protocol']
            model=xgb.Booster();model.load_model(dest/'model.ubj');model.set_param({'nthread':WORKERS})
            best=model.num_boosted_rounds()
        probabilities=model.inplace_predict(features[test_ids])
        predicted=probabilities.argmax(1)
        truth = targets[test_ids]
        np.savez_compressed(dest / 'predictions.npz', indices=test_ids, labels=truth,
                            probabilities=probabilities, fold=fold,
                            split='test', partial=False)
        accuracy = 100.0 * accuracy_score(truth, predicted)
        score = 100.0 * miou(truth, predicted)
        recall=100*float(np.mean([np.mean(predicted[truth==c]==c) if np.any(truth==c) else 0 for c in range(20)]))
        val_prob=model.inplace_predict(features[val_ids])
        np.savez_compressed(dest/'validation_predictions.npz',indices=val_ids,labels=targets[val_ids],probabilities=val_prob,fold=fold,split='val',partial=False)
        result=dict(protocol=protocol,best_round=best,seconds=time.time()-started,
                    execution=dict(system=platform.system(),machine=platform.machine(),python=platform.python_version(),numpy=np.__version__,xgboost=xgb.__version__,threads=WORKERS),
                    validation_miou=100*miou(targets[val_ids],val_prob.argmax(1)),
                    test=dict(oa=accuracy,miou=score,mean_recall=recall))
        if ACTION=='train':result_path.write_text(json.dumps(result,indent=2)+'\n')
        else:
            for key,value in result['test'].items():
                if not np.isclose(value,saved['test'][key],rtol=0,atol=1e-9):raise ValueError(f'Reloaded test metric differs: {key}')
            (dest/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
            result=saved
        records.append(result)
        print("FOLD_TEST {} {} oa {:.4f} miou {:.4f}".format(name, fold, accuracy, score), flush=True)
        del model,probabilities,val_prob;gc.collect()
    if len(records)==5:
        overall={k:dict(mean=float(np.mean([r['test'][k] for r in records])),sd=float(np.std([r['test'][k] for r in records],ddof=1))) for k in ['oa','miou','mean_recall']}
        (Path(OUT)/name/'overall.json').write_text(json.dumps(dict(folds=records,summary=overall),indent=2)+'\n')
        print('OVERALL',name,json.dumps(overall),flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    pids, targets = dataset_index()
    if FEATURES and (not np.array_equal(np.load(Path(FEATURES)/'targets.npy'),targets) or (Path(FEATURES,'pids.npy').exists() and not np.array_equal(np.load(Path(FEATURES)/'pids.npy').astype(str),np.asarray(pids).astype(str)))):
        raise ValueError('Feature-cache label order does not match dataset')
    n_samples = len(pids)
    print("PARCELS {}".format(n_samples), flush=True)
    fold_list = folds(n_samples)
    for fold, split in enumerate(fold_list, start=1):
        print('FOLD_SIZE',fold,{s:len(ids) for s,ids in split.items()},flush=True)
    np.save(os.path.join(OUT, "targets.npy"), targets)
    np.save(os.path.join(OUT, "pids.npy"),np.asarray(pids))

    path = os.path.join(FEATURES or OUT, "moments.npy")
    moments = None
    if os.path.isfile(path):
        try:
            probe = np.load(path, mmap_mode="r")
            if probe.shape == (n_samples, N_DATES, N_BANDS, N_MOMENTS) and probe.dtype == np.float32:
                moments = probe
                print("FEATURES cached {}".format(path), flush=True)
        except (ValueError, OSError):
            moments = None
    if moments is None:
        if FEATURES:raise ValueError('Requested feature cache is missing or incompatible; run prepare first')
        moments = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n_samples, N_DATES, N_BANDS, N_MOMENTS))
        data_dir = os.path.join(ROOT, "DATA")
        paths = [os.path.join(data_dir, "{}.npy".format(pid)) for pid in pids]
        chunks = [paths[start:start + CHUNK] for start in range(0, n_samples, CHUNK)]
        cursor = 0
        done = 0
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for block in pool.map(compute_chunk, chunks):
                moments[cursor:cursor + len(block)] = block
                cursor += len(block)
                done += 1
                if done % 10 == 0 or done == len(chunks):
                    moments.flush()
                    print("FEATURES {}/{}".format(cursor, n_samples), flush=True)
        moments.flush()

    if ACTION == "prepare":
        return

    sets = [
        ("mean_std", [0, 1]),
        ("mean_std_kurtosis", [0, 1, 3]),
        ("mean_std_skew_kurtosis_fifth", [0, 1, 2, 3, 4]),
    ]
    for name, which in sets:
        if name not in SETS:continue
        train_set(name, flatten(moments, which), targets, fold_list)
    print("DONE", flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','evaluate'])
    parser.add_argument('--dataset',default=ROOT);parser.add_argument('--output',default=OUT)
    parser.add_argument('--workers',type=int,default=WORKERS)
    parser.add_argument('--features',help='Existing read-only moments cache')
    parser.add_argument('--sets',nargs='+',default=['mean_std','mean_std_kurtosis'])
    parser.add_argument('--fold',type=int,choices=range(1,6));parser.add_argument('--rounds',type=int,default=200)
    args=parser.parse_args();ROOT=args.dataset;OUT=args.output;WORKERS=args.workers;ACTION=args.action;FEATURES=args.features;SETS=args.sets;FOLD=args.fold;ROUNDS=args.rounds
    main()

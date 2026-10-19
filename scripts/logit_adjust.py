"""Post-hoc logit adjustment on a frozen baseline checkpoint.

Menon et al., arXiv:2007.07314, eq. (9): predict argmax (z - tau * log pi),
where pi is the class frequency on that fold's training split.
"""

import os

if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pickle as pkl

import numpy as np
import torch
import torch.utils.data as data

from dataset import PixelSetData
from learning.metrics import mIou
from models.stclassifier import PseTae, get_ntrainparams
from train import SUB_CLASSES, configure_runtime, get_loaders, seed_worker


def main(args):
    torch.set_num_threads(4)
    config = {
        'rdm_seed': 1,
        'device': 'cpu',
        'kfold': 5,
        'batch_size': 128,
        'num_workers': 0,
    }
    configure_runtime(config)
    folder = args.dataset
    mean_std = pkl.load(open(folder + '/S2-2017-T31TFM-meanstd.pkl', 'rb'))
    dt = PixelSetData(folder, labels='label_44class', npixel=64, sub_classes=SUB_CLASSES,
                      norm=mean_std, extra_feature='geomfeat', return_id=False)
    loaders = get_loaders(dt, 5, config)
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    taus = np.round(np.linspace(0.0, 2.5, 51), 4)
    summary = []

    for fold, (train_loader, val_loader, test_loader, train_indices) in enumerate(loaders, start=1):
        del train_loader, test_loader
        counts = np.bincount([int(dt.target[i]) for i in train_indices], minlength=20).astype(np.float64)
        pi = counts / counts.sum()
        print('FOLD {} train {} val {} pi_min {:.6f}'.format(
            fold, len(train_indices), len(val_loader.sampler.indices), pi.min()), flush=True)

        model = PseTae(
            input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128],
            with_extra=True, extra_size=4, n_head=4, d_k=32, mlp3=[512, 128, 128],
            dropout=0.2, T=1000, len_max_seq=24, positions=dt.date_positions,
            mlp4=[128, 64, 32, 20])
        n_params = get_ntrainparams(model)
        if n_params != 164116:
            raise SystemExit('unexpected parameter count {}'.format(n_params))
        ckpt = torch.load(
            str(Path(args.baseline) / f'Fold_{fold}' / 'model.pth.tar'),
            map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        val_indices = list(val_loader.sampler.indices)
        loader = data.DataLoader(
            data.Subset(dt, val_indices), batch_size=128, shuffle=False, num_workers=2,
            worker_init_fn=seed_worker)
        logits = []
        labels = []
        with torch.no_grad():
            for step, (x, y) in enumerate(loader, start=1):
                logits.append(model(x).cpu())
                labels.append(y.cpu())
                if step % 50 == 0:
                    print('FOLD {} batch {}/{}'.format(fold, step, len(loader)), flush=True)
        logits = torch.cat(logits, dim=0).numpy().astype(np.float32)
        labels = torch.cat(labels, dim=0).numpy().astype(np.int64)
        np.savez(os.path.join(out_dir, 'fold_{}_logits.npz'.format(fold)),
                 logits=logits, labels=labels, pi=pi, counts=counts, val_indices=np.array(val_indices))

        log_pi = np.log(pi).astype(np.float32)
        curve = []
        for tau in taus:
            pred = np.argmax(logits - np.float32(tau) * log_pi, axis=1)
            oa = 100.0 * float(np.mean(pred == labels))
            miou = 100.0 * mIou(labels, pred, 20)
            curve.append({'tau': float(tau), 'oa': oa, 'miou': miou})
            print('CURVE {} tau {:.2f} oa {:.4f} miou {:.4f}'.format(fold, tau, oa, miou), flush=True)
        summary.append({
            'fold': fold,
            'checkpoint_epoch': int(ckpt['epoch']),
            'n_train': int(len(train_indices)),
            'n_val': int(len(labels)),
            'counts': counts.tolist(),
            'pi': pi.tolist(),
            'curve': curve,
        })
        with open(os.path.join(out_dir, 'curves.json'), 'w') as handle:
            json.dump(summary, handle)
        print('FOLD {} saved'.format(fold), flush=True)

    print('DONE', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='data/S2-2017-T31TFM-PixelSet')
    parser.add_argument('--baseline',default='results/baseline')
    parser.add_argument('--output',default='results/logit_adjust')
    main(parser.parse_args())

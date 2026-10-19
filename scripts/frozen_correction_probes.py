"""Zero-initialized logit corrections to the cached, frozen original decoder.

Same branches, splits, loss and optimizer as the preceding trajectory probes.
Epoch zero is verified against the cached baseline and is eligible for selection.
Fold 1 has been inspected previously: this is exploratory, not fresh confirmation.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from scripts import frozen_trajectory_probes as base


class CorrectionInputs(base.Inputs):
    def __init__(self, args, variant, norm=None):
        super().__init__(args, variant, norm)
        self.baseline_logits = self.data['logits']

    def batch(self, ids, device):
        batch, y = super().batch(ids, device)
        batch['baseline_logits'] = torch.from_numpy(self.baseline_logits[ids]).to(device)
        return batch, y


class Correction(base.Probe):
    def __init__(self, variant, days):
        super().__init__(variant, days)
        # Only the last layer is zeroed: hidden layers must retain nonzero random
        # initialization so the correction can begin learning immediately.
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, e, baseline_logits, raw=None, coords=None, mask=None, static=None):
        return baseline_logits.detach() + super().forward(e, raw, coords, mask, static)


def fit(args):
    folder = Path(args.output) / f'Fold_{args.fold}' / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    extraction = json.loads((Path(args.cache) / f'fold_{args.fold}_extraction.json').read_text())
    protocol = dict(variant=args.variant, fold=args.fold, seed=args.seed,
                    epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                    lr=args.lr, weight_decay=1e-4,
                    loss='Repository focal gamma=1, detached probability, no class weights',
                    baseline=extraction, trajectory_manifest_sha256=None,
                    prediction='Cached frozen baseline logits + trainable residual logits',
                    initialization='Final correction linear weights and bias zero; hidden layers random',
                    selection='Best validation mIoU, including verified baseline-equivalent epoch zero. Test only after every fit.',
                    caveat='Same previously inspected fold 1: exploratory evidence, requires independent-fold confirmation.')
    if args.variant != 'control':
        cache = args.patch_trajectories if args.variant in ['patch_temporal', 'spatial_nopos', 'spatial'] else args.trajectories
        protocol['trajectory_manifest_sha256'] = hashlib.sha256((Path(cache) / 'manifest.json').read_bytes()).hexdigest()
    if (folder / 'protocol.json').exists():
        assert json.loads((folder / 'protocol.json').read_text()) == protocol, 'Protocol mismatch'
    base.write_json(folder / 'protocol.json', protocol)
    if (folder / 'fit_complete.json').exists():
        print('FIT ALREADY COMPLETE', args.variant, flush=True)
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    inputs = CorrectionInputs(args, args.variant)
    model = Correction(args.variant, inputs.data['days']).to(args.device)
    tr, va = inputs.splits['train'], inputs.splits['val']
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=4)
    rng = np.random.RandomState(args.seed)
    start = time.time()
    best_epoch, history = 0, []

    def save(epoch, val, logits):
        artifact = dict(state_dict=model.state_dict(), normalization=inputs.norm,
                        protocol=protocol, epoch=epoch, val=val)
        torch.save(artifact, folder / 'best.tmp')
        (folder / 'best.tmp').replace(folder / 'best.pt')
        np.savez_compressed(folder / 'val_predictions.npz', logits=logits,
                            labels=inputs.y[va], indices=va, pids=inputs.data['pids'][va])

    logits = base.predict(model, inputs, va, args)
    assert np.array_equal(logits, inputs.baseline_logits[va]), 'Initial model differs from baseline'
    val = base.metrics(inputs.y[va], logits.argmax(1))
    best = val['miou']
    save(0, val, logits)
    history.append(dict(epoch=0, val=val, baseline_logits_identical=True))
    base.write_json(folder / 'history.json', history)
    scheduler.step(best)
    print('INIT', args.variant, 'baseline logits identical', 'val_miou', best,
          'parameters', sum(p.numel() for p in model.parameters()), flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        order = rng.permutation(tr)
        loss_sum = 0.
        for offset in range(0, len(order), args.batch_size):
            ids = order[offset:offset+args.batch_size]
            batch, y = inputs.batch(ids, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = base.focal_loss(model(**batch), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += loss.item() * len(ids)
        logits = base.predict(model, inputs, va, args)
        val = base.metrics(inputs.y[va], logits.argmax(1))
        scheduler.step(val['miou'])
        history.append(dict(epoch=epoch, train_loss=loss_sum/len(tr), val=val,
                            seconds=time.time()-epoch_start, lr=optimizer.param_groups[0]['lr']))
        base.write_json(folder / 'history.json', history)
        if val['miou'] > best:
            best, best_epoch = val['miou'], epoch
            save(epoch, val, logits)
        print(args.variant, 'epoch', epoch, 'val', round(100*val['oa'],3), round(100*val['miou'],3),
              'best', best_epoch, round(100*best,3), 'seconds', round(time.time()-epoch_start,1), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    base.write_json(folder / 'fit_complete.json', dict(best_epoch=best_epoch, val_miou=best,
                    epochs=epoch, seconds=time.time()-start,
                    parameters=sum(p.numel() for p in model.parameters()), baseline_initialization_verified=True))


def evaluate(args):
    base.evaluate(args, model_class=Correction, input_class=CorrectionInputs)
    folder = Path(args.output) / f'Fold_{args.fold}'
    result = json.loads((folder / 'results.json').read_text())
    result['method'] = 'Frozen original decoder plus zero-initialized trainable correction; epoch zero eligible.'
    result['caveat'] = 'Exploratory reuse of previously inspected fold 1. Parcel bootstrap conditions on this fold and trained models; independent folds/seeds needed for confirmation.'
    base.write_json(folder / 'results.json', result)
    report = folder / 'report.html'
    html = report.read_text().replace('Frozen trajectory probes', 'Frozen decoder correction probes')
    html = html.replace('Frozen baseline + pixel trajectories:', 'Frozen decoder + learned corrections:')
    note = '<p><strong>Original decoder stays frozen.</strong> All models begin with exactly the cached baseline logits; the final residual layer is initialized to zero. Validation may select epoch zero. This reuses the previously inspected fold and is exploratory, not independent confirmation.</p>'
    html = html.replace('<h2>Paired mIoU differences</h2>', note+'<h2>Paired mIoU differences</h2>')
    report.write_text(html)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['fit', 'evaluate'])
    p.add_argument('--cache', default='results/pair_probes')
    p.add_argument('--trajectories', default='results/frozen_trajectories/native_cache')
    p.add_argument('--patch-trajectories', default='results/frozen_trajectories/patch_cache')
    p.add_argument('--output', default='results/frozen_corrections')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--variant', choices=base.VARIANTS, default='control')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    globals()[args.command](args)


if __name__ == '__main__':
    main()

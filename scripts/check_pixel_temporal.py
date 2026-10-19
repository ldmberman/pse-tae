"""Checks that distinguish temporal pixels / unordered spatial attention from padding."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.pixel_temporal import PixelTemporalClassifier
from scripts.train_pixel_temporal import Parcels, EpochSampler, parser, runtime


def main():
    runtime(1)
    temporal = PixelTemporalClassifier(list(range(24)))
    runtime(1)
    spatial = PixelTemporalClassifier(list(range(24)), True)
    for key, value in temporal.state_dict().items():
        assert torch.equal(value, spatial.state_dict()[key]), key
    x = torch.randn(3, 7, 24, 10)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1]*7, [0]*7], dtype=torch.bool)
    geom = torch.randn(3, 4)
    for model in (temporal, spatial):
        model.eval()
        with torch.no_grad():
            original = model(x, mask, geom)
            assert torch.isfinite(original).all()
            order = torch.randperm(7)
            torch.testing.assert_close(original, model(x[:, order], mask[:, order], geom), atol=1e-5, rtol=1e-5)
            dirty = x.clone()
            dirty[~mask] = 1e5
            torch.testing.assert_close(original, model(dirty, mask, geom), atol=1e-5, rtol=1e-5)
            # Destroy trajectories while preserving every per-date pixel set.
            rearranged = x.clone()
            rearranged[1, :, 1::2] = x[1, torch.arange(6, -1, -1), 1::2]
            assert (original[1] - model(rearranged, mask, geom)[1]).abs().max() > 1e-5
        # Padding must also be excluded from training-time BN statistics.
        a, b = deepcopy(model).train(), deepcopy(model).train()
        torch.manual_seed(8)
        ya = a(x, mask, geom)
        torch.manual_seed(8)
        yb = b(dirty, mask, geom)
        torch.testing.assert_close(ya, yb)
        for k, val in a.state_dict().items():
            torch.testing.assert_close(val, b.state_dict()[k])
        ya.square().mean().backward()
        assert torch.isfinite(a.spectral[0].weight.grad).all()
        assert a.spectral[0].weight.grad.abs().sum() > 0
        assert a.temporal.attention_heads.fc1_k.weight.grad.abs().sum() > 0
        if a.spatial is not None:
            assert a.spatial.self_attn.in_proj_weight.grad.abs().sum() > 0
        singleton = torch.zeros_like(mask)
        singleton[0, 0] = True
        assert torch.isfinite(a(x, singleton, geom)).all()
        assert torch.isfinite(a(x, torch.zeros_like(mask), geom)).all()
    args = parser().parse_args(['train', '--device', 'cpu'])
    ds, ev = Parcels(args, True), Parcels(args)
    idx = int(ds.splits['train'][0])
    first, again = ds[(idx, 1)], ds[(idx, 1)]
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], ds[(idx, 2)][0])
    np.testing.assert_array_equal(first[2], ev[idx][2])
    s1, s2 = EpochSampler(ds.splits['train'], 1), EpochSampler(ds.splits['train'], 1)
    s1.epoch = s2.epoch = 5
    assert list(s1) == list(s2)
    # Match cached evaluation to full native trajectories, not independently sampled dates.
    from scripts.frozen_trajectory_probes import sample_native
    raw = np.load(Path(args.native) / 'DATA' / f'{ds.pids[idx]}.npy')
    sampled, _, valid, _ = sample_native(raw, ds.pids[idx])
    np.testing.assert_array_equal(sampled, ev.pixels[idx])
    np.testing.assert_array_equal(valid, ev.mask[idx])
    print('PASS: shared initialization, pixel permutation, train/eval padding, trajectory sensitivity, gradients, empty/singleton pixels, reproducible sampling, exact baseline evaluation pixels', flush=True)


if __name__ == '__main__':
    main()

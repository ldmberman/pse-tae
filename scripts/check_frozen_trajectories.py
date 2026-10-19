"""Scientific invariants for the frozen trajectory experiment (no pytest needed)."""
import numpy as np
import torch
from frozen_trajectory_probes import (Probe, TrajectoryEncoder, sample_patch,
                                      sample_native, static_features, partitions, focal_loss, VARIANTS)

torch.set_num_threads(2)
torch.manual_seed(5)
rng = np.random.RandomState(5)
days = np.arange(24) * 12
dates = np.array([20170101 + (i // 2) * 100 for i in range(20)] + [20171015] * 4)
# Replace synthetic dates with valid month bins; temporal positions are separate.
dates = np.array([20170001 + (1 + i * 9 // 23) * 100 for i in range(24)])

a = np.zeros((24, 10, 32, 32), np.uint16)
first = rng.randint(100, 5000, (24, 10)).astype(np.uint16)
second = rng.randint(100, 5000, (24, 10)).astype(np.uint16)
a[:, :, 0, 0] = first
a[:, :, 0, 1] = first
a[:, :, 20, 20] = second
x, xy, mask, count = sample_patch(a, 23)
assert count == 2 and mask.sum() == 2
assert np.array_equal(x, sample_patch(a, 23)[0])
s = static_features(x, mask, days, dates)
assert s.shape == (880,) and np.isfinite(s).all()
order = rng.permutation(64)
assert np.allclose(s, static_features(x[order], mask[order], days, dates))
bad_padding = x.copy()
bad_padding[~mask] = 60000
assert np.array_equal(s, static_features(bad_padding, mask, days, dates))
assert np.array_equal(static_features(x, np.zeros_like(mask), days, dates), np.zeros(880))
native = rng.randint(0, 5000, (24,10,100)).astype(np.uint16)
native_x, _, native_mask, _ = sample_native(native, 23)
baseline_indices = np.random.RandomState(20261013 + 23).choice(list(range(100)), size=64, replace=False)
assert np.array_equal(native_x, native[:,:,baseline_indices].transpose(2,0,1))
assert native_mask.all()
constant = np.full((2,24,10), 1000, np.uint16)
constant[1] = 3000
summaries = static_features(constant, np.ones(2,bool), days, dates).reshape(5,11,16)
np.testing.assert_allclose(summaries[0,0,:10], .2, atol=1e-6)
np.testing.assert_allclose(summaries[1,0,:10], .1, atol=1e-6)
np.testing.assert_allclose(summaries[2,0,:10], .12, atol=1e-6)
np.testing.assert_allclose(summaries[4,0,:10], .28, atol=1e-6)
np.testing.assert_allclose(summaries[:,3,:], 0, atol=1e-6)  # amplitude
np.testing.assert_allclose(summaries[:,5:7,:], 0, atol=1e-6)  # slopes
np.testing.assert_allclose(summaries[:,:,10:], 0, atol=1e-6)  # equal-band indices
split = partitions(191412, 1)
assert len(set(np.concatenate(list(split.values())))) == 191412
assert not (set(split['train']) & set(split['val']))

raw = torch.tensor(rng.randint(100, 5000, (3, 6, 24, 10)), dtype=torch.float32)
coords = torch.randn(3, 6, 2)
valid = torch.tensor([[1,1,1,1,0,0],[1,1,1,1,1,1],[0,0,0,0,0,0]], dtype=torch.bool)
perm = torch.tensor([3,1,5,2,0,4])
for spatial, positional in [(False, False), (True, False), (True, True)]:
    m = TrajectoryEncoder(days, spatial, positional).eval()
    z = m(raw, coords, valid)
    assert torch.isfinite(z).all(), 'Empty parcel produced nonfinite features'
    zp = m(raw[:,perm], coords[:,perm], valid[:,perm])
    torch.testing.assert_close(z, zp, atol=1e-5, rtol=1e-5)
    changed = raw.clone()
    changed[~valid] = 60000
    torch.testing.assert_close(z, m(changed, coords, valid), atol=1e-5, rtol=1e-5)
    shuffled_coords = m(raw, coords[:,perm], valid)
    if positional:
        assert not torch.allclose(z, shuffled_coords), 'Spatial encoder ignored layout'
    else:
        torch.testing.assert_close(z, shuffled_coords)

for variant in VARIANTS:
    m = Probe(variant, days)
    e = torch.randn(3,128)
    logits = m(e, raw, coords, valid, torch.randn(3,880))
    loss = focal_loss(logits, torch.tensor([1,4,19]))
    loss.backward()
    assert torch.isfinite(loss) and logits.shape == (3,20)
    assert not e.requires_grad
    if variant != 'control':
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.branch.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
print('PASS: splits, duplicate handling, static formulas/masking, permutation invariance, spatial sensitivity, empty parcels, and branch gradients')

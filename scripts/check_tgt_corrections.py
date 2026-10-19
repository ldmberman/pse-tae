"""Checks grouping geometry, preserved widths, masking, and residual learning."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.tgt_correction import GroupedCorrection, moments
from scripts.tgt_correction_probes import group_patch
from scripts.frozen_trajectory_probes import focal_loss


def main():
    torch.set_num_threads(2)
    spatial_ids = np.arange(1, 1025, dtype=np.uint16).reshape(32, 32)
    a = np.broadcast_to(spatial_ids, (24, 10, 32, 32)).copy()
    a[:, :, 0, 0] = 0
    groups, mask = group_patch(a)
    for r in range(4):
        for c in range(4):
            np.testing.assert_array_equal(groups[r*4+c, :, 0, 0], a[0, 0, r*8:(r+1)*8, c*8:(c+1)*8].ravel())
    np.testing.assert_array_equal(groups[..., 0, 0], groups[..., -1, -1])
    assert mask.sum() == 1023 and not mask[0, 0]
    x = torch.tensor([[[1.], [3.], [999.]]])
    m = moments(x, torch.tensor([[True, True, False]]), 1)
    torch.testing.assert_close(m, torch.tensor([[2., 2.**.5]]))
    days, mean, std = list(range(24)), np.zeros((24, 10)), np.ones((24, 10))
    torch.manual_seed(1)
    temporal = GroupedCorrection(days, mean, std, 'temporal')
    torch.manual_seed(1)
    spatial = GroupedCorrection(days, mean, std, 'spatial_nopos')
    for k, v in temporal.state_dict().items():
        assert torch.equal(v, spatial.state_dict()[k]), k
    torch.manual_seed(4)
    raw = torch.randn(3, 4, 5, 24, 10)
    valid = torch.ones(3, 4, 5, dtype=torch.bool)
    valid[0, 0, 2:] = False
    valid[0, 3] = False
    valid[2] = False
    e = torch.randn(3, 128, requires_grad=True)
    logits = torch.randn(3, 20, requires_grad=True)
    labels = torch.tensor([1, 4, 18])
    for model in [GroupedCorrection(days, mean, std, 'control'), temporal, spatial]:
        model.eval()
        batch = dict(e=e, baseline_logits=logits, raw=raw, mask=valid)
        assert torch.equal(model(**batch), logits.detach())
        if model.variant != 'control':
            assert model.temporal.in_channels == 128
            assert model.temporal.attention_heads.d_k == 32
            assert model.temporal.n_neurons == [512, 128, 128]
            with torch.no_grad():
                out = model.group_encodings(raw, valid)
                assert out.shape == (3, 256) and (out[2] == 0).all()
                dirty = raw.clone()
                dirty[~valid] = 1e5
                torch.testing.assert_close(out, model.group_encodings(dirty, valid))
                order = torch.tensor([2, 0, 3, 1])
                torch.testing.assert_close(out, model.group_encodings(raw[:, order], valid[:, order]), atol=2e-5, rtol=2e-5)
                perm = torch.tensor([4, 0, 2, 1, 3])
                torch.testing.assert_close(out, model.group_encodings(raw[:, :, perm], valid[:, :, perm]), atol=2e-5, rtol=2e-5)
                changed = raw.clone()
                changed[1, :, :, 1::2] = raw[1, order, :, 1::2]
                assert (out[1] - model.group_encodings(changed, valid)[1]).abs().max() > 1e-5
        model.train()
        assert torch.equal(model(**batch), logits.detach())
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        focal_loss(model(**batch), labels).backward()
        assert e.grad is None and logits.grad is None
        assert model.classifier[-1].weight.grad.abs().sum() > 0
        opt.step()
        opt.zero_grad()
        focal_loss(model(**batch), labels).backward()
        if model.variant != 'control':
            assert model.spectral[0].weight.grad.abs().sum() > 0
            assert model.temporal.attention_heads.fc1_k.weight.grad.abs().sum() > 0
            if model.spatial is not None:
                assert model.spatial.self_attn.in_proj_weight.grad.abs().sum() > 0
            singleton = torch.zeros_like(valid)
            singleton[0, 0, 0] = True
            assert torch.isfinite(model.group_encodings(raw, singleton)).all()
            assert (model.group_encodings(raw, torch.zeros_like(valid)) == 0).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        print('PASS', model.variant, 'parameters', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    print('PASS grouping, N-1 std, shared initialization, full widths, zero corrections, masks, permutations, temporal sensitivity, frozen inputs, gradients', flush=True)


if __name__ == '__main__':
    main()

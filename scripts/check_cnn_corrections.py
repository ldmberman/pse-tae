"""Check grid reconstruction, locality, zero correction, masks and gradients."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.cnn_correction import CNNCorrection, restore_grid, VARIANTS
from scripts.tgt_correction_probes import group_patch
from scripts.frozen_trajectory_probes import focal_loss


def main():
    torch.set_num_threads(2)
    a = np.arange(24*10*32*32).reshape(24, 10, 32, 32).astype(np.uint16)
    grouped, mask = group_patch(a)
    grid, support = restore_grid(torch.from_numpy(grouped[None].astype(np.float32)), torch.from_numpy(mask[None]))
    np.testing.assert_array_equal(grid[0].numpy(), a)
    assert support.all()
    torch.manual_seed(1)
    raw = torch.randn(3, 16, 64, 24, 10)
    mask = torch.ones(3, 16, 64, dtype=torch.bool)
    mask[0, 0] = False
    mask[2] = False
    e, logits = torch.randn(3, 128, requires_grad=True), torch.randn(3, 20, requires_grad=True)
    for variant in VARIANTS:
        torch.manual_seed(1)
        model = CNNCorrection(list(range(24)), np.zeros((24,10)), np.ones((24,10)), variant)
        model.eval()
        batch = dict(e=e, baseline_logits=logits, raw=raw, mask=mask)
        assert torch.equal(model(**batch), logits.detach())
        if variant != 'control':
            with torch.no_grad():
                out = model.group_encodings(raw, mask)
                assert out.shape == (3,256) and (out[2] == 0).all()
                dirty = raw.clone()
                dirty[~mask] = 1e5
                torch.testing.assert_close(out, model.group_encodings(dirty, mask))
                # Same multiset in each cell, different arrangement: CNN must
                # respond to this, unlike independent-pixel set pooling.
                perm = torch.randperm(64)
                shuffled = model.group_encodings(raw[:,:,perm], mask[:,:,perm])
                assert (out[:2] - shuffled[:2]).abs().max() > 1e-4
                if variant == 'cnn_temporal':
                    assert model.temporal.in_channels == 128
                    assert model.temporal.attention_heads.d_k == 32
                else:
                    torch.testing.assert_close(out, model.group_encodings(raw.flip(3), mask), atol=1e-5, rtol=1e-5)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        for step in range(2):
            opt.zero_grad()
            loss = focal_loss(model(**batch), torch.tensor([1,4,18]))
            assert torch.isfinite(loss)
            loss.backward()
            assert e.grad is None and logits.grad is None
            assert model.classifier[-1].weight.grad.abs().sum() > 0
            if step and variant != 'control':
                assert model.conv1.weight.grad.abs().sum() > 0
                if variant == 'cnn_temporal':
                    assert model.temporal.attention_heads.fc1_k.weight.grad.abs().sum() > 0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
        print('PASS', variant, 'trainable', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    print('PASS exact grid reconstruction, local spatial sensitivity, masking, frozen inputs and residual learning')


if __name__ == '__main__':
    main()

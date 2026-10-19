"""Check baseline identity, frozen logits, and gradient flow after zero initialization."""
import numpy as np
import torch
from frozen_correction_probes import Correction
from frozen_trajectory_probes import VARIANTS, focal_loss

torch.set_num_threads(2)
for variant in VARIANTS:
    torch.manual_seed(13)
    m = Correction(variant, np.arange(24)*12)
    frozen = torch.randn(4,20,requires_grad=True)
    original = frozen.detach().clone()
    inputs = dict(e=torch.randn(4,128), baseline_logits=frozen,
                  raw=torch.rand(4,6,24,10)*5000, coords=torch.rand(4,6,2),
                  mask=torch.tensor([[1,1,1,1,0,0],[1,1,1,1,1,1],[0,0,0,0,0,0],[1,0,0,0,0,0]],dtype=torch.bool),
                  static=torch.randn(4,880))
    y = torch.tensor([1,4,18,19])
    for training in [False, True]:
        m.train(training)
        assert torch.equal(m(**inputs), original), 'Zero correction changed logits'
    optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)
    optimizer.zero_grad()
    focal_loss(m(**inputs),y).backward()
    assert frozen.grad is None
    assert m.classifier[-1].weight.grad.abs().sum()>0
    # Expected: earlier layers see zero gradient until the output weights move.
    assert m.embedding[0].weight.grad.abs().sum()==0
    optimizer.step()
    optimizer.zero_grad()
    logits = m(**inputs)
    assert not torch.equal(logits,original)
    focal_loss(logits,y).backward()
    assert m.embedding[0].weight.grad.abs().sum()>0
    if variant!='control':
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.branch.parameters())
    assert frozen.grad is None and torch.equal(frozen.detach(),original)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    print('PASS',variant,'exact baseline initialization; frozen logits; correction and branch learn',flush=True)

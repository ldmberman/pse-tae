"""Check matched inputs, original encoder parity, and end-to-end gradients."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models.tgt_scratch import GroupedScratch, parameter_count, VARIANTS
from scripts.frozen_trajectory_probes import focal_loss


def main():
    torch.set_num_threads(2)
    mean, std = np.zeros((24,10),np.float32), np.ones((24,10),np.float32)
    models={}
    for v in VARIANTS:
        torch.manual_seed(1)
        models[v]=GroupedScratch(list(range(24)),mean,std,v)
    counts={v:parameter_count(m) for v,m in models.items()}
    assert abs(counts['capacity']/counts['grouped']-1)<.01
    for part in ['spatial_encoder','temporal_encoder']:
        a=getattr(models['baseline'].base,part).state_dict()
        b=getattr(models['grouped'].base,part).state_dict()
        assert all(torch.equal(a[k],b[k]) for k in a),part
    raw=torch.randn(3,16,3,24,10)
    mask=torch.ones(3,16,3,dtype=torch.bool)
    geom=torch.randn(3,4)
    mask[2]=False
    for v,m in models.items():
        m.eval()
        with torch.no_grad():
            logits=m(raw,mask,geom)
            assert logits.shape==(3,20) and torch.isfinite(logits).all()
            permutation=torch.randperm(16)
            torch.testing.assert_close(logits,m(raw[:,permutation],mask[:,permutation],geom),atol=2e-5,rtol=2e-5)
            changed=raw.clone()
            changed[:2,:,:,1::2]=raw[:2,permutation,:,1::2]
            alternative=m(changed,mask,geom)
            if v=='grouped':
                assert (logits-alternative).abs().max()>1e-4
                assert m.groups(raw,mask).shape==(3,256)
                assert m.base.temporal_encoder.in_channels==128
                assert m.base.decoder[0].in_features==384
            else:
                torch.testing.assert_close(logits,alternative,atol=2e-5,rtol=2e-5)
        m.train()
        loss=focal_loss(m(raw,mask,geom),torch.tensor([0,4,18]))
        loss.backward()
        for name,p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(),(v,name)
        assert m.base.spatial_encoder.mlp1[0].lin.weight.grad.abs().sum()>0
        if v=='grouped':
            assert m.groups.spectral[0].weight.grad.abs().sum()>0
            assert m.groups.temporal.attention_heads.fc1_k.weight.grad.abs().sum()>0
    # Baseline forward is exactly the unmodified original PSE+TAE on these inputs.
    m=models['baseline'].eval()
    x=raw[:2].reshape(2,48,24,10).permute(0,2,3,1).contiguous()
    masks=mask[:2].reshape(2,48)[:,None,:].expand(2,24,48).contiguous().float()
    g=geom[:2,None,:].expand(2,24,4).contiguous()
    with torch.no_grad():
        torch.testing.assert_close(m(raw[:2],mask[:2],geom[:2]),m.base(((x,masks),(g,None))),rtol=0,atol=0)
    print('PASS: counts',counts,'; original-model parity, shared initialization, dimensions, permutation behavior, missing patches, and gradients')


if __name__=='__main__':
    main()

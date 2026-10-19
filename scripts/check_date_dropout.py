"""Ensure date removal preserves checkpoint behavior and acquisition positions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import numpy as np
import torch
from models.date_dropout import DateDropoutCapacity
from models.tgt_scratch import GroupedScratch
from scripts.frozen_trajectory_probes import focal_loss


def main():
    torch.set_num_threads(2)
    days = [3+13*i for i in range(24)]
    mean = np.arange(240,dtype=np.float32).reshape(24,10)/100
    std = np.ones((24,10),np.float32)*2
    torch.manual_seed(1)
    original = GroupedScratch(days,mean,std,'capacity')
    model = DateDropoutCapacity(days,mean,std)
    model.load_state_dict(original.state_dict(),strict=True)
    raw,mask,geometry = torch.randn(3,16,4,24,10),torch.ones(3,16,4,dtype=torch.bool),torch.randn(3,4)
    mask[2] = False
    original.eval();model.eval()
    with torch.no_grad():
        assert torch.equal(original(raw,mask,geometry),model(raw,mask,geometry))
    # Same seed restores ordinary internal dropout. Full index list must match
    # original forward even in training, including batch-normalization updates.
    a,b = copy.deepcopy(model).train(),copy.deepcopy(model).train()
    torch.manual_seed(12); expected=a(raw,mask,geometry)
    torch.manual_seed(12); actual=b(raw,mask,geometry,torch.arange(24))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    ix = torch.tensor([0,3,5,8,12,17,23])
    a,b = copy.deepcopy(model).train(),copy.deepcopy(model).train()
    dirty=raw.clone(); omitted=sorted(set(range(24))-set(ix.tolist()))
    dirty[:,:,:,omitted,:]=1e8
    torch.manual_seed(15); expected=a(raw,mask,geometry,ix)
    torch.manual_seed(15); actual=b(dirty,mask,geometry,ix)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    # Compare with a physically shortened original model supplied actual days
    # and matching normalization rows; this independently checks positional IDs.
    shortened=GroupedScratch([days[i] for i in ix],mean[ix],std[ix],'capacity')
    state=model.state_dict().copy()
    state['mean']=state['mean'][ix];state['std']=state['std'][ix]
    state['base.temporal_encoder.position_enc.weight']=state['base.temporal_encoder.position_enc.weight'][ix]
    shortened.load_state_dict(state,strict=True);shortened.train()
    a=copy.deepcopy(model).train()
    torch.manual_seed(22);actual=a(raw,mask,geometry,ix)
    torch.manual_seed(22);expected=shortened(raw[:,:,:,ix,:],mask,geometry)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    loss=focal_loss(actual,torch.tensor([1,3,18]));loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters())
    assert a.base.spatial_encoder.mlp1[0].lin.weight.grad.abs().sum()>0
    print('PASS checkpoint compatibility, all-date equivalence, excluded-date invariance, actual-day positional encodings, finite end-to-end gradients')


if __name__=='__main__':
    main()

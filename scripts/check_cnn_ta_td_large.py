"""Check date identities, masking, initialization, and end-to-end gradients."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy
import numpy as np
import torch
from models.cnn_ta_td_large import CNNTATDLarge
from models.capacity_large import CapacityLarge
from models.tgt_scratch import parameter_count
from scripts.frozen_trajectory_probes import focal_loss


def main():
    torch.set_num_threads(2)
    days=[3+13*i for i in range(24)]
    mean=np.arange(240,dtype=np.float32).reshape(24,10)/100
    std=np.ones((24,10),np.float32)*2
    torch.manual_seed(1);control=CapacityLarge(days,mean,std)
    torch.manual_seed(1);model=CNNTATDLarge(days,mean,std)
    for name,tensor in control.state_dict().items():
        if name.startswith(('base.spatial_encoder.','base.temporal_encoder.')):
            assert torch.equal(tensor,model.state_dict()[name]),name
    raw=torch.randn(3,16,64,24,10)
    mask=torch.rand(3,16,64)>.3;mask[2]=False
    geometry=torch.randn(3,4)
    a,b=copy.deepcopy(model).train(),copy.deepcopy(model).train()
    torch.manual_seed(12);expected=a(raw,mask,geometry)
    torch.manual_seed(12);actual=b(raw,mask,geometry,torch.arange(24))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    ix=torch.tensor([0,3,5,8,12,17,23])
    a,b=copy.deepcopy(model).train(),copy.deepcopy(model).train()
    dirty=raw.clone();dirty[:,:,:,sorted(set(range(24))-set(ix.tolist())),:]=1e6
    dirty[~mask]=1e6
    torch.manual_seed(15);expected=a(raw,mask,geometry,ix)
    torch.manual_seed(15);actual=b(dirty,mask,geometry,ix)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    shortened=CNNTATDLarge([days[i] for i in ix],mean[ix],std[ix])
    state=model.state_dict().copy()
    for name in ('mean','std','base.temporal_encoder.position_enc.weight','temporal.position_enc.weight'):
        state[name]=state[name][ix]
    shortened.load_state_dict(state,strict=True);shortened.train()
    a=copy.deepcopy(model).train()
    torch.manual_seed(22);actual=a(raw,mask,geometry,ix)
    torch.manual_seed(22);expected=shortened(raw[:,:,:,ix,:],mask,geometry)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    loss=focal_loss(actual,torch.tensor([1,3,18]));loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters())
    for prefix in ('base.spatial_encoder','base.temporal_encoder','conv1','conv2','projection','temporal','base.decoder'):
        assert sum(float(p.grad.abs().sum()) for name,p in a.named_parameters() if name.startswith(prefix) and p.grad is not None)>0,prefix
    # Preserve the same parcel spectrum multiset while changing local layout.
    model.eval()
    perm=torch.randperm(1024)
    rearranged=raw.reshape(3,1024,24,10)[:,perm].reshape_as(raw)
    shuffled_mask=mask.reshape(3,1024)[:,perm].reshape_as(mask)
    with torch.no_grad():
        assert not torch.allclose(model(raw,mask,geometry)[:2],model(rearranged,shuffled_mask,geometry)[:2],atol=1e-5,rtol=1e-5)
    print('PASS: matched base initialization; exact all-date equivalence; excluded dates/masked pixels; original date positions in BOTH branches; finite gradients through every branch; layout sensitivity; missing-patch finite outputs')
    print('Trainable parameters:',parameter_count(model))


if __name__=='__main__':main()

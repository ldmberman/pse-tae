"""Matched patch-input PSE+TAE models, trained end to end from scratch."""
import torch
from torch import nn

from learning.weight_init import weight_init
from models.stclassifier import PseTae
from models.tgt_correction import GroupedCorrection


VARIANTS = ('baseline', 'capacity', 'grouped')


class GroupTrajectoryEncoder(GroupedCorrection):
    def __init__(self, days, mean, std):
        super().__init__(days, mean, std, 'temporal')
        del self.classifier

    def forward(self, raw, mask):
        return self.group_encodings(raw, mask)


class GroupedScratch(nn.Module):
    def __init__(self, days, mean, std, variant='baseline', capacity_width=204):
        super().__init__()
        assert variant in VARIANTS
        self.variant = variant
        width = capacity_width if variant == 'capacity' else 128
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        decoder = ([384, 128, 64, 20] if variant == 'grouped' else
                   [width, 128, 64, 20] if variant == 'capacity' else [128, 64, 32, 20])
        # Isolate initialization so baseline/grouped shared encoders start identically.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed())
            self.base = PseTae(positions=list(days), mlp2=[132, width],
                              mlp3=[4 * width, width, width], mlp4=decoder)
            # Apply in module order: decoder shape cannot affect encoder initialization.
            for module in (self.base.spatial_encoder, self.base.temporal_encoder):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(torch.initial_seed())
                    module.apply(weight_init)
            self.base.decoder.apply(weight_init)
        self.groups = GroupTrajectoryEncoder(days, mean, std) if variant == 'grouped' else None

    def base_embedding(self, raw, mask, geometry):
        b, g, p, t, c = raw.shape
        x = ((raw.float() - self.mean) / self.std).reshape(b, g * p, t, c)
        valid = mask.reshape(b, g * p).clone()
        # Known missing patches have no pixels. Use one zero normalized dummy,
        # identically in all variants, so the original PSE pooling stays finite.
        empty = ~valid.any(1)
        x = x.masked_fill(empty[:, None, None, None], 0)
        valid[empty, 0] = True
        pixels = x.permute(0, 2, 3, 1).contiguous()
        masks = valid[:, None, :].expand(b, t, g * p).contiguous().float()
        extra = geometry[:, None, :].expand(b, t, 4).contiguous()
        # Explicit extra-feature tuple avoids the original PSE's len(extra)==2
        # ambiguity when a batch happens to contain exactly two parcels.
        return self.base.temporal_encoder(self.base.spatial_encoder(((pixels, masks), (extra, None))))

    def forward(self, raw, mask, geometry):
        z = self.base_embedding(raw, mask, geometry)
        if self.groups is not None:
            z = torch.cat([z, self.groups(raw, mask)], 1)
        return self.base.decoder(z)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

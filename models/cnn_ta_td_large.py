"""Large PSE+TAE plus local CNN trajectories; train-time physical date removal."""
import torch
from torch import nn
from models.capacity_large import CapacityLarge
from models.cnn_correction import restore_grid, CNNCorrection
from models.tgt_correction import moments
from models.tae import TemporalAttentionEncoder
from models.decoder import get_decoder
from learning.weight_init import weight_init


def attend_dates(tae, x, indices):
    """Keep the original acquisition-day sinusoid for every surviving date."""
    if indices is None:
        return tae(x)
    assert tae.inconv is None and tae.pos_encoding == 'sin'
    z = tae.inlayernorm(x) + tae.position_enc(indices)[None]
    z, _ = tae.attention_heads(z, z, z)
    z = z.permute(1, 0, 2).contiguous().view(len(x), -1)
    return tae.outlayernorm(tae.dropout(tae.mlp(z)))


class CNNTATDLarge(CapacityLarge):
    def __init__(self, days, mean, std):
        super().__init__(days, mean, std)
        self.conv1 = nn.Conv2d(10, 32, 3, padding=1)
        self.norm1 = nn.LayerNorm(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2 = nn.LayerNorm(64)
        self.projection = nn.Sequential(nn.Linear(128, 384), nn.LayerNorm(384), nn.ReLU())
        self.temporal = TemporalAttentionEncoder(in_channels=384, n_head=4, d_k=32,
            n_neurons=[1536, 384, 384], positions=list(days), dropout=.2)
        self.branch_norm = nn.LayerNorm(768)
        # Preserve both representations: base384 + CNN trajectory mean/std768.
        self.base.decoder = get_decoder([1152, 256, 128, 20])
        for module in (self.conv1, self.conv2, self.projection, self.temporal, self.base.decoder):
            module.apply(weight_init)

    def forward(self, raw, mask, geometry, date_indices=None):
        ix = None
        mean, std = self.mean, self.std
        if date_indices is not None:
            assert self.training, 'Validation/test must use all acquisition dates'
            ix = torch.as_tensor(date_indices, dtype=torch.long, device=raw.device)
            assert ix.ndim == 1 and len(ix) >= 2 and bool((ix[1:] > ix[:-1]).all())
            raw = raw.index_select(3, ix)
            mean, std = mean.index_select(0, ix), std.index_select(0, ix)
        b, g, p, t, c = raw.shape
        # Canonical zero background, matching the patch loader. Prevent masked
        # values from entering the PSE MLP batch statistics.
        raw = raw.masked_fill(~mask[..., None, None], 0)
        normalized = (raw.float() - mean) / std
        # Same large PSE+TAE pathway and missing-patch handling as the control.
        x = normalized.reshape(b, g*p, t, c)
        valid = mask.reshape(b, g*p).clone()
        empty = ~valid.any(1)
        x = x.masked_fill(empty[:, None, None, None], 0)
        valid[empty, 0] = True
        pixels = x.permute(0, 2, 3, 1).contiguous()
        masks = valid[:, None].expand(b, t, g*p).contiguous().float()
        extra = geometry[:, None].expand(b, t, 4).contiguous()
        base_dates = self.base.spatial_encoder(((pixels, masks), (extra, None)))
        base = attend_dates(self.base.temporal_encoder, base_dates, ix)
        # CNN features exchange context across cell boundaries (5x5 receptive field).
        grid, support = restore_grid(normalized, mask)
        grid = grid * support[:, None, None]
        m = support[:, None, None].expand(b, t, 1, 32, 32).reshape(b*t, 1, 32, 32)
        x = CNNCorrection.activate(self.conv1(grid.reshape(b*t, c, 32, 32)), self.norm1, m)
        x = CNNCorrection.activate(self.conv2(x), self.norm2, m)
        x = x.reshape(b, t, 64, 4, 8, 4, 8).permute(0, 3, 5, 4, 6, 1, 2)
        x = x.reshape(b, 16, 64, t, 64)
        summaries = moments(x, mask[..., None].expand(b, 16, 64, t), 2)
        present = mask.any(2)
        selected = summaries[present]
        z = summaries.new_zeros(b, 16, 384)
        if len(selected):
            n = len(selected)
            if self.training and n == 1:
                selected = selected.repeat(2, 1, 1)
            z[present] = attend_dates(self.temporal, self.projection(selected), ix)[:n]
        branch = self.branch_norm(moments(z, present, 1)) * present.any(1, keepdim=True)
        return self.base.decoder(torch.cat([base, branch], 1))

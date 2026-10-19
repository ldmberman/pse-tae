"""Capacity PSE+TAE with physical date removal and original date embeddings.

No parameter names/shapes change. All-date forward delegates to the original
model; omitted dates are removed before the spatial encoder and query averaging.
"""
import torch
from models.tgt_scratch import GroupedScratch


class DateDropoutCapacity(GroupedScratch):
    def __init__(self, days, mean, std):
        super().__init__(days, mean, std, variant='capacity')

    def forward(self, raw, mask, geometry, date_indices=None):
        if date_indices is None:
            return super().forward(raw, mask, geometry)
        assert self.training, 'Evaluation must use every date'
        ix = torch.as_tensor(date_indices, dtype=torch.long, device=raw.device)
        assert ix.ndim == 1 and len(ix) >= 2
        assert bool((ix[1:] > ix[:-1]).all())
        b, g, p, _, c = raw.shape
        t = len(ix)
        x = ((raw.index_select(3, ix).float() - self.mean.index_select(0, ix)) /
             self.std.index_select(0, ix)).reshape(b, g*p, t, c)
        valid = mask.reshape(b, g*p).clone()
        empty = ~valid.any(1)
        x = x.masked_fill(empty[:, None, None, None], 0)
        valid[empty, 0] = True
        pixels = x.permute(0, 2, 3, 1).contiguous()
        masks = valid[:, None, :].expand(b, t, g*p).contiguous().float()
        extra = geometry[:, None, :].expand(b, t, 4).contiguous()
        z = self.base.spatial_encoder(((pixels, masks), (extra, None)))
        tae = self.base.temporal_encoder
        assert tae.inconv is None and tae.pos_encoding == 'sin' and tae.positions is not None
        # Row ix contains the original acquisition-day sinusoid. Do not replace
        # it with arange(t), which would relabel the surviving dates.
        z = tae.inlayernorm(z) + tae.position_enc(ix)[None]
        z, _ = tae.attention_heads(z, z, z)
        z = z.permute(1, 0, 2).contiguous().view(b, -1)
        z = tae.outlayernorm(tae.dropout(tae.mlp(z)))
        return self.base.decoder(z)

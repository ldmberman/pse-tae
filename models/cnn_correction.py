"""Local CNN patterns, optionally retaining sixteen temporal trajectories."""
import torch
from torch import nn
from models.tgt_correction import GroupedCorrection, moments
from models.tae import TemporalAttentionEncoder
from learning.weight_init import weight_init


VARIANTS = ['control', 'cnn_spatial', 'cnn_temporal']


def restore_grid(raw, mask):
    """Invert group_patch exactly: B,16,64,T,C -> B,T,C,32,32."""
    b, g, p, t, c = raw.shape
    assert (g, p) == (16, 64)
    grid = raw.reshape(b, 4, 4, 8, 8, t, c).permute(0, 5, 6, 1, 3, 2, 4)
    support = mask.reshape(b, 4, 4, 8, 8).permute(0, 1, 3, 2, 4)
    return grid.reshape(b, t, c, 32, 32), support.reshape(b, 32, 32)


class CNNCorrection(GroupedCorrection):
    def __init__(self, days, mean, std, variant='cnn_spatial'):
        assert variant in VARIANTS
        # Reuse the exact same 384-128-64-20 zero-initialized correction head.
        super().__init__(days, mean, std, 'control')
        self.variant = variant
        if variant == 'control':
            return
        # Normalization at each spatial location avoids padded cells affecting
        # batch statistics. Both convolutions preserve the original grid.
        self.conv1 = nn.Conv2d(10, 32, 3, padding=1)
        self.norm1 = nn.LayerNorm(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2 = nn.LayerNorm(64)
        self.branch_norm = nn.LayerNorm(256)
        self.conv1.apply(weight_init)
        self.conv2.apply(weight_init)
        if variant == 'cnn_temporal':
            self.temporal = TemporalAttentionEncoder(in_channels=128, n_head=4, d_k=32,
                n_neurons=[512, 128, 128], positions=list(days), dropout=.2)
            self.temporal.apply(weight_init)

    @staticmethod
    def activate(x, norm, support):
        x = torch.relu(norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x * support

    def group_encodings(self, raw, mask):
        grid, support = restore_grid(raw, mask)
        b, t, c, h, w = grid.shape
        grid = (grid - self.spectral_mean[None, :, :, None, None]) / self.spectral_std[None, :, :, None, None]
        grid = grid * support[:, None, None]
        m = support[:, None, None].expand(b, t, 1, h, w).reshape(b*t, 1, h, w)
        x = self.activate(self.conv1(grid.reshape(b*t, c, h, w)), self.norm1, m)
        x = self.activate(self.conv2(x), self.norm2, m)
        # Neighbouring cells may exchange local context through the 5x5
        # receptive field; cell membership itself remains fixed across dates.
        if self.variant == 'cnn_spatial':
            x = x.reshape(b, t, 64, h*w).permute(0, 1, 3, 2)
            valid = support.reshape(b, 1, h*w).expand(b, t, h*w)
            per_date = moments(x, valid, 2)  # B,T,128
            present = support.flatten(1).any(1)
            pooled = moments(per_date, present[:, None].expand(b, t), 1)
        else:
            x = x.reshape(b, t, 64, 4, 8, 4, 8).permute(0, 3, 5, 4, 6, 1, 2)
            x = x.reshape(b, 16, 64, t, 64)
            summaries = moments(x, mask[..., None].expand(b, 16, 64, t), 2)
            group_mask = mask.any(2)
            selected = summaries[group_mask]
            z = summaries.new_zeros(b, 16, 128)
            if len(selected):
                n = len(selected)
                if self.training and n == 1:
                    selected = selected.repeat(2, 1, 1)
                z[group_mask] = self.temporal(selected)[:n]
            pooled = moments(z, group_mask, 1)
            present = group_mask.any(1)
        return self.branch_norm(pooled) * present[:, None]

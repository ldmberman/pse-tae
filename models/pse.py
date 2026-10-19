"""
Pixel-Set encoder module

author: Vivien Sainte Fare Garnot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class PixelSetEncoder(nn.Module):
    def __init__(self, input_dim, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[64, 128], with_extra=True,
                 extra_size=4, kurtosis_stop_grad=False, parcel_stats='', prototype_distances=False,
                 raw_kurtosis_extra=False, seq_len=24, num_classes=20):
        """
        Pixel-set encoder.
        Args:
            input_dim (int): Number of channels of the input tensors
            mlp1 (list):  Dimensions of the successive feature spaces of MLP1
            pooling (str): Pixel-embedding pooling strategy. Any underscore-separated
                combination of mean, std, skew, kurtosis, max, and min. kurtosis is excess
                kurtosis (0 for a Gaussian), divided by kurtosis_gap when that buffer is set.
                With the paper MLP1 (64-d) and 4 geometric features, mean_std_kurtosis
                requires mlp2 to start at 196, and mean_std_skew_kurtosis at 260.
            mlp2 (list): Dimensions of the successive feature spaces of MLP2
            with_extra (bool): Whether additional pre-computed features are passed between the two MLPs
            extra_size (int, optional): Number of channels of the additional features, if any.
        """

        super(PixelSetEncoder, self).__init__()

        self.input_dim = input_dim
        self.mlp1_dim = copy.deepcopy(mlp1)
        self.mlp2_dim = copy.deepcopy(mlp2)
        self.pooling = pooling

        self.with_extra = with_extra
        self.extra_size = extra_size

        self.name = 'PSE-{}-{}-{}'.format('|'.join(list(map(str, self.mlp1_dim))), pooling,
                                          '|'.join(list(map(str, self.mlp2_dim))))

        self.output_dim = input_dim * len(pooling.split('_')) if len(self.mlp2_dim) == 0 else self.mlp2_dim[-1]

        inter_dim = self.mlp1_dim[-1] * len(pooling.split('_'))


        if self.with_extra:
            self.name += 'Extra'
            inter_dim += self.extra_size

        self.parcel_stats = parcel_stats or ''
        self.prototype_distances = bool(prototype_distances)
        self.raw_kurtosis_extra = bool(raw_kurtosis_extra)
        self.seq_len = int(seq_len)
        self.num_classes = int(num_classes)
        if self.parcel_stats == '':
            self.parcel_stat_dim = 0
        elif self.parcel_stats == 'mean_std':
            self.parcel_stat_dim = 2 * input_dim
        elif self.parcel_stats == 'mean_std_kurtosis':
            self.parcel_stat_dim = 3 * input_dim
        else:
            raise ValueError('Unknown parcel_stats {}'.format(self.parcel_stats))
        if self.prototype_distances:
            self.parcel_stat_dim += self.num_classes
        if self.raw_kurtosis_extra:
            inter_dim += input_dim

        assert mlp1[0] == input_dim + self.parcel_stat_dim
        assert (inter_dim == mlp2[0])
        self.kurtosis_stop_grad = bool(kurtosis_stop_grad)
        if 'kurtosis' in self.pooling.split('_'):
            self.register_buffer('kurtosis_gap', torch.ones(self.mlp1_dim[-1]))
        else:
            self.kurtosis_gap = None
        if self.parcel_stats == 'mean_std_kurtosis' or self.raw_kurtosis_extra:
            self.register_buffer('raw_kurtosis_gap', torch.ones(input_dim))
        else:
            self.raw_kurtosis_gap = None
        if self.prototype_distances:
            self.register_buffer('prototypes', torch.zeros(self.num_classes, self.seq_len, 2 * input_dim))
        else:
            self.prototypes = None
        # Feature extraction
        layers = []
        for i in range(len(self.mlp1_dim) - 1):
            layers.append(linlayer(self.mlp1_dim[i], self.mlp1_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        # MLP after pooling
        layers = []
        for i in range(len(self.mlp2_dim) - 1):
            layers.append(nn.Linear(self.mlp2_dim[i], self.mlp2_dim[i + 1]))
            layers.append(nn.BatchNorm1d(self.mlp2_dim[i + 1]))
            if i < len(self.mlp2_dim) - 2:
                layers.append(nn.ReLU())
        self.mlp2 = nn.Sequential(*layers)

    def forward(self, input):
        """
        The input of the PSE is a tuple of tensors as yielded by the PixelSetData class:
          (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
        Pixel-Set : Batch_size x (Sequence length) x Channel x Number of pixels
        Pixel-Mask : Batch_size x (Sequence length) x Number of pixels
        Extra-features : Batch_size x (Sequence length) x Number of features

        If the input tensors have a temporal dimension, it will be combined with the batch dimension so that the
        complete sequences are processed at once. Then the temporal dimension is separated back to produce a tensor of
        shape Batch_size x Sequence length x Embedding dimension
        """
        pixels, mask, extra, reshape_needed, batch, temp = self._prepare(input)
        out = self.mlp1(self._with_parcel_stats(pixels, mask, temp))
        parts = []
        for name in self.pooling.split('_'):
            stat = pooling_methods[name](out, mask)
            if name == 'kurtosis':
                stat = stat / self.kurtosis_gap
                if self.kurtosis_stop_grad:
                    stat = stat.detach()
            parts.append(stat)
        out = torch.cat(parts, dim=1)

        if self.raw_kurtosis_extra:
            raw_k = masked_kurtosis(pixels, mask) / self.raw_kurtosis_gap
            extra = raw_k if extra is None else torch.cat([extra, raw_k], dim=1)
        if extra is not None:
            out = torch.cat([out, extra], dim=1)
        out = self.mlp2(out)

        if reshape_needed:
            out = out.view(batch, temp, -1)
        return out

    def date_mean_kurtosis(self, input):
        """Channel-mean excess kurtosis of the MLP1 embeddings, one value per date."""
        pixels, mask, _, reshape_needed, batch, temp = self._prepare(input)
        with torch.no_grad():
            hidden = self.mlp1(pixels)
            kurtosis = masked_kurtosis(hidden, mask).mean(dim=1)
        if reshape_needed:
            kurtosis = kurtosis.view(batch, temp)
        return kurtosis

    def raw_kurtosis(self, input):
        """Excess kurtosis of the reflectance across pixels, before MLP1."""
        pixels, mask, _, _, _, _ = self._prepare(input)
        return masked_kurtosis(pixels, mask)

    def _with_parcel_stats(self, pixels, mask, temp):
        if self.parcel_stat_dim == 0:
            return pixels
        mean = masked_mean(pixels, mask)
        std = masked_std(pixels, mask)
        width = pixels.shape[-1]

        def expand(value):
            return value.unsqueeze(-1).expand(-1, -1, width)

        parts = [pixels, expand(mean), expand(std)]
        if self.parcel_stats == 'mean_std_kurtosis':
            parts.append(expand(masked_kurtosis(pixels, mask) / self.raw_kurtosis_gap))
        if self.prototypes is not None:
            parts.append(expand(self._prototype_distances(mean, std, temp)))
        return torch.cat(parts, dim=1)

    def _prototype_distances(self, mean, std, temp):
        values = torch.cat([mean, std], dim=1)
        period = self.seq_len if temp is None else int(temp)
        dates = torch.arange(values.shape[0], device=values.device) % period
        centers = self.prototypes[:, dates, :]
        distances = torch.linalg.norm(values.unsqueeze(0) - centers, dim=-1)
        return distances.transpose(0, 1)

    def embedding_kurtosis(self, input):
        """Raw excess kurtosis of the MLP1 embeddings, before the percentile rescaling."""
        out, mask, _, _, _, _ = self._prepare(input)
        out = self.mlp1(out)
        return masked_kurtosis(out, mask)

    def _prepare(self, input):
        a, b = input
        if isinstance(a, (tuple, list)) and len(a) == 2:
            out, mask = a
            extra = b
            if isinstance(extra, (tuple, list)) and len(extra) == 2:
                extra, _ = extra
        else:
            out, mask = a, b
            extra = None

        reshape_needed = False
        batch = temp = None
        if len(out.shape) == 4:
            reshape_needed = True
            batch, temp = out.shape[:2]
            out = out.view(batch * temp, *out.shape[2:])
            mask = mask.view(batch * temp, -1)
            if extra is not None:
                extra = extra.view(batch * temp, -1)
        return out, mask, extra, reshape_needed, batch, temp

class linlayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, input):
        out = input.permute((0, 2, 1))  # to channel last
        out = self.lin(out)

        out = out.permute((0, 2, 1))  # to channel first
        out = self.bn(out)
        out = F.relu(out)

        return out

def masked_mean(x, mask):
    out = x.permute((1, 0, 2))
    out = out * mask
    out = out.sum(dim=-1) / mask.sum(dim=-1)
    out = out.permute((1, 0))
    return out

def masked_std(x, mask):
    m = masked_mean(x, mask)

    out = x.permute((2, 0, 1))
    out = out - m
    out = out.permute((2, 1, 0))

    out = out * mask
    d = mask.sum(dim=-1)
    d[d == 1] = 2

    out = (out ** 2).sum(dim=-1) / (d - 1)
    out = torch.sqrt(out + 10e-32) # To ensure differentiability
    out = out.permute(1, 0)
    return out

def _centered(x, mask, mean):
    centered = x - mean.unsqueeze(-1)
    return centered * mask.unsqueeze(1)


def masked_skew(x, mask):
    """Standardized third moment across pixels. Zero when fewer than 3 pixels are real."""
    mean = masked_mean(x, mask)
    std = masked_std(x, mask)
    d = mask.sum(dim=-1)
    m3 = (_centered(x, mask, mean) ** 3).sum(dim=-1) / d.clamp(min=1).unsqueeze(1)
    skew = m3 / std.clamp(min=1e-6).pow(3)
    return torch.where(d.unsqueeze(1) >= 3, skew, torch.zeros_like(skew))


def masked_kurtosis(x, mask):
    """Excess kurtosis across pixels. Zero when fewer than 4 pixels are real."""
    mean = masked_mean(x, mask)
    std = masked_std(x, mask)
    d = mask.sum(dim=-1)
    m4 = (_centered(x, mask, mean) ** 4).sum(dim=-1) / d.clamp(min=1).unsqueeze(1)
    kurt = m4 / std.clamp(min=1e-6).pow(4) - 3
    return torch.where(d.unsqueeze(1) >= 4, kurt, torch.zeros_like(kurt))


def maximum(x, mask):
    return x.max(dim=-1)[0].squeeze()

def minimum(x, mask):
    return x.min(dim=-1)[0].squeeze()

pooling_methods = {
    'mean': masked_mean,
    'std': masked_std,
    'skew': masked_skew,
    'kurtosis': masked_kurtosis,
    'max': maximum,
    'min': minimum
}

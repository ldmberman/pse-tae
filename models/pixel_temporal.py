"""Encode each native pixel through time, then aggregate the unordered pixel set."""
import torch
from torch import nn

from models.tae import TemporalAttentionEncoder
from models.decoder import get_decoder
from learning.weight_init import weight_init


class PixelTemporalClassifier(nn.Module):
    def __init__(self, days, spatial=False):
        super().__init__()
        self.spectral = nn.Sequential(nn.Linear(10, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                      nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.temporal = TemporalAttentionEncoder(
            in_channels=64, n_head=4, d_k=16, n_neurons=[256, 64, 64],
            positions=list(days), len_max_seq=len(days), dropout=.2)
        self.parcel = nn.Sequential(nn.Linear(132, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.decoder = get_decoder([128, 64, 32, 20])
        # Initialize all shared parameters before constructing the optional block.
        self.apply(weight_init)
        self.spatial = (nn.TransformerEncoderLayer(
            64, 4, dim_feedforward=128, dropout=.2, batch_first=True,
            norm_first=True, activation='relu') if spatial else None)

    def forward(self, pixels, mask, geometry):
        b, p, t, c = pixels.shape
        valid = pixels[mask]  # Exclude padding before all temporal batch norms.
        encoded = pixels.new_zeros(b, p, 64)
        if len(valid):
            n = len(valid)
            # The original TAE's query BN requires >=2 examples when training.
            if self.training and n == 1:
                valid = valid.repeat(2, 1, 1)
            features = self.spectral(valid.reshape(-1, c)).reshape(len(valid), t, 64)
            encoded[mask] = self.temporal(features)[:n]
        if self.spatial is not None:
            safe_mask = mask.clone()
            safe_mask[~mask.any(1), 0] = True  # Avoid all-masked attention softmax.
            encoded = self.spatial(encoded, src_key_padding_mask=~safe_mask)
        weights = mask[..., None].to(encoded.dtype)
        count = weights.sum(1).clamp_min(1)
        mean = (encoded * weights).sum(1) / count
        variance = ((encoded - mean[:, None]).square() * weights).sum(1) / count
        std = (variance + 1e-6).sqrt()
        # Empty parcels contribute geometry only; no arbitrary padding signature.
        nonempty = mask.any(1, keepdim=True)
        mean, std = mean * nonempty, std * nonempty
        return self.decoder(self.parcel(torch.cat([mean, std, geometry], dim=1)))

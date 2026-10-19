"""Full-width TGT encodings added to frozen PSE+TAE logits."""
import torch
from torch import nn
from models.tae import TemporalAttentionEncoder
from learning.weight_init import weight_init


def moments(x, mask, dim):
    """Masked mean and sample std; same N-1 convention as the original PSE."""
    w = mask.to(x.dtype).unsqueeze(-1)
    n = w.sum(dim)
    mean = (x * w).sum(dim) / n.clamp_min(1)
    var = ((x - mean.unsqueeze(dim)).square() * w).sum(dim) / (n - 1).clamp_min(1)
    std = (var + 1e-31).sqrt()
    return torch.cat([mean, std], -1) * (n > 0)


class GroupedCorrection(nn.Module):
    def __init__(self, days, mean, std, variant='temporal'):
        super().__init__()
        assert variant in ['control', 'temporal', 'spatial_nopos']
        self.variant = variant
        self.register_buffer('spectral_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('spectral_std', torch.as_tensor(std, dtype=torch.float32))
        # Fork RNG so every variant receives exactly the same correction head.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(107)
            self.classifier = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Dropout(.2),
                                            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 20))
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)
        if variant != 'control':
            self.spectral = nn.Sequential(nn.Linear(10, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                          nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU())
            self.temporal = TemporalAttentionEncoder(in_channels=128, n_head=4, d_k=32,
                n_neurons=[512, 128, 128], positions=list(days), dropout=.2)
            self.spectral.apply(weight_init)
            self.temporal.apply(weight_init)
            self.branch_norm = nn.LayerNorm(256)
        # Instantiate last to keep shared encoder weights identical across variants.
        self.spatial = (nn.TransformerEncoderLayer(128, 4, dim_feedforward=256,
            dropout=.2, batch_first=True, norm_first=True, activation='relu')
            if variant == 'spatial_nopos' else None)

    def group_encodings(self, raw, mask):
        # raw: B,16,64,24,10; mask: B,16,64. Grid membership is fixed over dates.
        b, g, p, t, _ = raw.shape
        selected = raw[mask].float()
        features = selected.new_zeros((b, g, p, t, 64))
        if len(selected):
            selected = (selected - self.spectral_mean) / self.spectral_std
            features[mask] = self.spectral(selected.reshape(-1, 10)).reshape(-1, t, 64)
        group_mask = mask.any(2)
        summaries = moments(features, mask[..., None].expand(b, g, p, t), 2)
        valid = summaries[group_mask]
        z = summaries.new_zeros(b, g, 128)
        if len(valid):
            n = len(valid)
            if self.training and n == 1:
                valid = valid.repeat(2, 1, 1)
            z[group_mask] = self.temporal(valid)[:n]
        if self.spatial is not None:
            safe = group_mask.clone()
            safe[~safe.any(1), 0] = True
            z = self.spatial(z, src_key_padding_mask=~safe)
        pooled = moments(z, group_mask, 1)
        return self.branch_norm(pooled) * group_mask.any(1, keepdim=True)

    def forward(self, e, baseline_logits, raw=None, mask=None):
        summary = e.new_zeros(len(e), 256) if self.variant == 'control' else self.group_encodings(raw, mask)
        delta = self.classifier(torch.cat([e.detach(), summary], 1))
        return baseline_logits.detach() + delta

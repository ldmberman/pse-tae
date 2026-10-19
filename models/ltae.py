"""
Lightweight Temporal Attention Encoder (Garnot and Landrieu, arXiv:2007.00586).

This matches the released training code. One linear map produces every head's
keys from the full embedding. Each head's values are its own channel group,
and its master query is a learned parameter.
"""

import sys

import numpy as np
import torch
import torch.nn as nn

from models.tae import get_sinusoid_encoding_table


PAPER_LTAE = {
    'n_head': 16,
    'd_k': 8,
    'mlp3': [256, 128],
    'd_model': 256,
}


def apply_ltae_defaults(config, argv=None):
    """Fill the Table 1 configuration for flags the caller did not set."""
    if argv is None:
        argv = sys.argv[1:]
    for key, value in PAPER_LTAE.items():
        if '--' + key not in argv:
            config[key] = value
    if config.get('lms') is None:
        config['lms'] = 24


class LTAE(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=None, dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False):
        super(LTAE, self).__init__()
        if n_neurons is None:
            n_neurons = [256, 128]
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = list(n_neurons)
        self.return_att = return_att
        self.len_max_seq = len_max_seq

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(
                nn.Conv1d(in_channels, d_model, 1),
                nn.LayerNorm((d_model, len_max_seq)))
        else:
            self.d_model = in_channels
            self.inconv = None

        if self.d_model % n_head != 0:
            raise ValueError('d_model {} is not divisible by n_head {}'.format(self.d_model, n_head))
        if self.n_neurons[0] != self.d_model:
            raise ValueError('mlp3 must start at d_model {}, got {}'.format(self.d_model, self.n_neurons[0]))

        if positions is None:
            table_positions = len_max_seq + 1
        else:
            table_positions = positions
        group_width = self.d_model // n_head
        sin_tab = get_sinusoid_encoding_table(table_positions, group_width, T=T)
        self.position_enc = nn.Embedding.from_pretrained(
            torch.cat([sin_tab for _ in range(n_head)], dim=1),
            freeze=True)

        self.inlayernorm = nn.LayerNorm(self.in_channels)
        self.outlayernorm = nn.LayerNorm(self.n_neurons[-1])
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([
                nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                nn.BatchNorm1d(self.n_neurons[i + 1]),
                nn.ReLU()])
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        sz_b, seq_len, _ = x.shape
        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positions is None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long, device=x.device).expand(sz_b, seq_len)
        else:
            src_pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).expand(sz_b, seq_len)
        enc_output = x + self.position_enc(src_pos)

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)
        enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)
        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        if self.return_att:
            return enc_output, attn
        return enc_output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_k, d_in):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.empty(n_head, d_k))
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / d_k))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / d_k))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        query = self.Q.unsqueeze(1).expand(n_head, sz_b, d_k).reshape(-1, d_k)
        keys = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)

        groups = torch.stack(v.split(d_in // n_head, dim=-1))
        values = groups.view(n_head * sz_b, seq_len, -1)
        output, attn = self.attention(query, keys, values)

        attn = attn.view(n_head, sz_b, 1, seq_len).squeeze(dim=2)
        output = output.view(n_head, sz_b, 1, d_in // n_head).squeeze(dim=2)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

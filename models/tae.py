"""
Temporal Attention Encoder module

Credits:
The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.

paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""

import torch
import torch.nn as nn
import numpy as np
import copy


class TemporalAttentionEncoder(nn.Module):
    def __init__(self, in_channels=128, n_head=4, d_k=32, d_model=None, n_neurons=[512, 128, 128], dropout=0.2,
                 T=1000, len_max_seq=24, positions=None, pos_encoding='sin', rope_base=10000.0):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
            pos_encoding (str): 'sin' adds the fixed sinusoid. 'rope' rotates queries and keys instead.
            rope_base (float): RoPE frequency base. Used only when pos_encoding is 'rope'.

        """

        super(TemporalAttentionEncoder, self).__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.pos_encoding = pos_encoding
        self.n_neurons = copy.deepcopy(n_neurons)

        if pos_encoding == 'rope':
            base_label = int(rope_base) if float(rope_base).is_integer() else rope_base
            self.name = 'TAE_dk{}_{}Heads_{}_rope{}_do{}'.format(
                d_k, n_head, '|'.join(list(map(str, self.n_neurons))), base_label, dropout)
        else:
            self.name = 'TAE_dk{}_{}Heads_{}_T{}_do{}'.format(d_k, n_head, '|'.join(list(map(str, self.n_neurons))), T,
                                                              dropout)

        if positions is None:
            table_positions = len_max_seq + 1
        else:
            table_positions = positions
            self.name += '_bespokePos'

        if pos_encoding == 'sin':
            self.position_enc = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(table_positions, self.in_channels, T=T),
                freeze=True)
        elif pos_encoding == 'rope':
            self.position_enc = None
        else:
            raise ValueError("pos_encoding must be 'sin' or 'rope', got {!r}".format(pos_encoding))

        self.inlayernorm = nn.LayerNorm(self.in_channels)

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(nn.Conv1d(in_channels, d_model, 1),
                                        nn.LayerNorm(d_model, len_max_seq))
            self.name += '_dmodel{}'.format(d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.outlayernorm = nn.LayerNorm(self.d_model)

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model,
            rope_base=rope_base if pos_encoding == 'rope' else None)

        assert (self.n_neurons[0] == n_head * self.d_model)
        assert (self.n_neurons[-1] == self.d_model)
        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                           nn.BatchNorm1d(self.n_neurons[i + 1]),
                           nn.ReLU()])

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        sz_b, seq_len, d = x.shape

        x = self.inlayernorm(x)

        if self.pos_encoding == 'sin':
            if self.positions is None:
                src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
            else:
                src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
            enc_output = x + self.position_enc(src_pos)
            rope_pos = None
        else:
            enc_output = x
            rope_pos = self._rope_positions(seq_len, x.device)

        if self.inconv is not None:
            enc_output = self.inconv(enc_output.permute(0, 2, 1)).permute(0, 2, 1)

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output, rope_pos)

        enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)  # Concatenate heads

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        return enc_output

    def _rope_positions(self, seq_len, device):
        if self.positions is None:
            values = list(range(seq_len))
        else:
            values = list(self.positions)[:seq_len]
            if len(values) != seq_len:
                raise ValueError('RoPE expected {} positions, got {}'.format(seq_len, len(self.positions)))
        return torch.tensor(values, device=device, dtype=torch.float32)


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_in, rope_base=None):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.rope = RotaryEmbedding(d_k, base=rope_base) if rope_base is not None else None

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(n_head * d_k),
            nn.Linear(n_head * d_k, n_head * d_k)
        )

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v, positions=None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = self.fc1_q(q).view(sz_b, seq_len, n_head, d_k)
        k = self.fc1_k(k).view(sz_b, seq_len, n_head, d_k)

        if self.rope is None:
            q = q.mean(dim=1).squeeze()  # MEAN query
            q = self.fc2(q.view(sz_b, n_head * d_k)).view(sz_b, n_head, d_k)
        else:
            # Project each date first. Rotation then commutes with the mean, so a
            # global shift of the dates leaves the attention weights unchanged.
            q = self.fc2(q.reshape(sz_b * seq_len, n_head * d_k)).view(sz_b, seq_len, n_head, d_k)
            q, k = self.rope(q, k, positions)
            q = q.mean(dim=1)

        q = q.permute(1, 0, 2).contiguous().view(n_head * sz_b, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = v.repeat(n_head, 1, 1)  # (n*b) x lv x d_in

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, 1, d_in)
        output = output.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
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


class RotaryEmbedding(nn.Module):
    """Rotary position embedding on the last dimension. No learned parameters."""

    def __init__(self, dim, base=10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError('RoPE requires an even dimension, got {}'.format(dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, q, k, positions):
        cos, sin = self._cos_sin(positions)
        # (1, T, 1, D) broadcasts over batch and heads.
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

    def _cos_sin(self, positions):
        freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    ''' Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)'''

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # Stay on CPU here. PseTae.to(device) moves the frozen embedding, including onto an L40.
    return torch.FloatTensor(sinusoid_table)

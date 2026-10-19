import torch
import torch.nn as nn

from models.pse import PixelSetEncoder
from models.tae import TemporalAttentionEncoder
from models.ltae import LTAE
from models.decoder import get_decoder


class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4, kurtosis_stop_grad=False, parcel_stats='', prototype_distances=False,
                 raw_kurtosis_extra=False, seq_len=24, num_classes=20, date_kurtosis=False,
                 n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000, len_max_seq=24,
                 positions=None, pos_encoding='sin', rope_base=10000.0,
                 mlp4=[128, 64, 32, 20]):
        super(PseTae, self).__init__()
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size, kurtosis_stop_grad=kurtosis_stop_grad,
                                               parcel_stats=parcel_stats, prototype_distances=prototype_distances,
                                               raw_kurtosis_extra=raw_kurtosis_extra, seq_len=seq_len,
                                               num_classes=num_classes)
        self.date_kurtosis = bool(date_kurtosis)
        if self.date_kurtosis:
            self.register_buffer('date_kurtosis_gap', torch.ones(1))
        self.temporal_encoder = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=len_max_seq, positions=positions,
                                                         pos_encoding=pos_encoding, rope_base=rope_base)
        self.decoder = get_decoder(mlp4)
        self.name = '_'.join([self.spatial_encoder.name, self.temporal_encoder.name])

    def forward(self, input):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        out = self.spatial_encoder(input)
        out = self.temporal_encoder(out)
        if self.date_kurtosis:
            kurtosis = self.spatial_encoder.date_mean_kurtosis(input) / self.date_kurtosis_gap
            out = torch.cat([out, kurtosis], dim=1)
        out = self.decoder(out)
        return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))


class PseLTae(nn.Module):
    """Pixel-set encoder with the lightweight temporal attention encoder."""

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4, n_head=16, d_k=8, d_model=256, mlp3=[256, 128], dropout=0.2, T=1000,
                 len_max_seq=24, positions=None, mlp4=[128, 64, 32, 20], return_att=False):
        super(PseLTae, self).__init__()
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.temporal_encoder = LTAE(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                     n_neurons=mlp3, dropout=dropout, T=T, len_max_seq=len_max_seq,
                                     positions=positions, return_att=return_att)
        self.decoder = get_decoder(mlp4)
        self.return_att = return_att
        self.name = '_'.join([self.spatial_encoder.name, 'LTAE'])

    def forward(self, input):
        out = self.spatial_encoder(input)
        if self.return_att:
            out, att = self.temporal_encoder(out)
            return self.decoder(out), att
        out = self.temporal_encoder(out)
        return self.decoder(out)

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)
        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(
            s / total * 100, t / total * 100, c / total * 100))


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

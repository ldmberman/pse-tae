"""Larger capacity control with the same pixel features and attention heads."""
import torch
from models.tgt_scratch import GroupedScratch
from models.decoder import get_decoder
from learning.weight_init import weight_init


class CapacityLarge(GroupedScratch):
    def __init__(self, days, mean, std):
        super().__init__(days, mean, std, variant='capacity', capacity_width=384)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed())
            self.base.decoder = get_decoder([384, 256, 128, 20])
            self.base.decoder.apply(weight_init)

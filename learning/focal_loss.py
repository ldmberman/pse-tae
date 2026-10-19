"""
Credits to  github.com/clcarwin/focal_loss_pytorch
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float)
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1).long()

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        if self.alpha is not None:
            if self.alpha.device != input.device or self.alpha.dtype != input.dtype:
                self.alpha = self.alpha.to(device=input.device, dtype=input.dtype)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class LogitAdjustedLoss(nn.Module):
    """Cross-entropy on logits shifted by tau * log(class prior).

    Menon et al., arXiv:2007.07314, equation (10). Prediction stays argmax of
    the raw logits; the prior offset is only inside the loss.
    """

    def __init__(self, tau, log_prior):
        super(LogitAdjustedLoss, self).__init__()
        self.tau = float(tau)
        self.log_prior = torch.tensor(log_prior, dtype=torch.float)

    def forward(self, logits, target):
        log_prior = self.log_prior.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits + self.tau * log_prior, target.long())

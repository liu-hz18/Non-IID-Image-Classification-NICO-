import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional

from .mlp import MLPClassifier
from .attention import AttentionMLP

SQRT_2 = np.sqrt(2)
PI2 = 2 * np.pi

def rff(x: torch.Tensor):
    w = torch.randn_like(x)
    phi = torch.rand_like(x) * PI2
    return SQRT_2 * torch.cos(w * x + phi)


class SampleWeighting(nn.Module):

    def __init__(self, bsz, alpha:float=0.1):
        super(SampleWeighting, self).__init__()
        self.weight = nn.Parameter(torch.ones(bsz).type(torch.float32))
        self.alpha = alpha
    
    def reset_parameters(self):
        nn.init.constant_(self.weight, 1.0)

    def forward(self, features: torch.Tensor):
        n, m = features.shape
        weight = torch.abs(self.weight[:n])
        rff_feat = rff(features)
        weighted_rff = (rff_feat.T * weight).T
        mean_rff = weighted_rff.mean(dim=0)
        shifted_rff = weighted_rff - mean_rff
        shifted_rff_T = torch.transpose(shifted_rff, 0, 1)
        rff_cov = shifted_rff_T.mm(shifted_rff)
        loss = rff_cov.sum() / (n-1) + self.alpha * ((weight - 1.0)**2).sum()
        # print(loss, weight, rff_cov.sum() / (n-1))
        return loss

    def apply_weight(self, loss: torch.Tensor):
        n = loss.shape[0]
        weight = torch.abs(self.weight[:n])
        weight /= weight.mean()
        return (loss * weight).mean()

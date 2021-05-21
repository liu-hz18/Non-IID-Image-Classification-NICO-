import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional

from .mlp import MLPClassifier


class AttentionMLP(nn.Module):

    def __init__(self, num_features: int, num_classes: int, hidden_states: int = None, droprate: float = 0.5):
        super(AttentionMLP, self).__init__()
        self.intermediate = num_features*3
        self.mlp1 = MLPClassifier(
            num_features, num_features, self.intermediate, droprate
        )
        self.mlp2 = MLPClassifier(
            num_features, num_features, self.intermediate, droprate
        )
        self.mlp3 = MLPClassifier(
            num_features, num_classes, None, droprate
        )

    def forward(self, features):
        mask = torch.sigmoid(self.mlp1(features))
        new_feat = mask * features # attention
        x = self.mlp2(new_feat) + new_feat  # shortcut
        output = self.mlp3(x)
        return output

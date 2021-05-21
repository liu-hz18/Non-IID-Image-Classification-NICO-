import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional


class MLPClassifier(nn.Module):

    def __init__(self, num_features: int, num_classes: int, hidden_states:int=None, droprate:float=0.5):
        super(MLPClassifier, self).__init__()
        if hidden_states is None:
            hidden_states = num_features * 4
        self.linear1 = nn.Linear(num_features, hidden_states)
        self.dropout = nn.Dropout(droprate)
        self.linear2 = nn.Linear(hidden_states, num_classes)

    def forward(self, features):
        x = self.dropout(F.relu(self.linear1(features)))
        output = self.linear2(x)
        return output

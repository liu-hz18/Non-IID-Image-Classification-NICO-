import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional

from .attention import AttentionMLP
from .mlp import MLPClassifier


class Discriminator(nn.Module):

    def __init__(self, num_features: int, num_classes: int):
        super(Discriminator, self).__init__()
        self.domain_classifier = MLPClassifier(num_features, num_classes)

    def forward(self, x):
        output = self.domain_classifier(x)
        return output


class DiscriminatorStack(nn.Module):

    def __init__(self, num_features: int, num_classes: int, num_domains: int):
        super(DiscriminatorStack, self).__init__()
        self.domain_classifiers = nn.ModuleList([
            Discriminator(num_features, num_domains)
            for _ in range(num_classes)
        ])

    def forward(self, feature: torch.Tensor, cls_idx: int):
        domain_logits = self.domain_classifiers[cls_idx](feature)
        return domain_logits


class GANN(nn.Module):

    def __init__(self, num_features: int, num_classes: int, num_domains: int, droprate: float=0.5, alpha: float=0.1):
        super(GANN, self).__init__()
        self.feature_extractor = AttentionMLP(num_features, num_features, droprate=droprate)
        self.classifier_mlp1 = MLPClassifier(num_features, num_features, droprate=droprate)
        self.classifier_norm = nn.BatchNorm1d(num_features)
        self.classifier_mlp2 = MLPClassifier(num_features, num_classes, droprate=droprate)

    def forward(self, feature: torch.Tensor, return_feature: bool=False):
        feat = self.feature_extractor(feature)
        x = self.classifier_mlp1(feat)
        output = self.classifier_norm(x + feat) # add and norm
        class_logits = self.classifier_mlp2(output)
        if return_feature:
            return class_logits, feat
        else:
            return class_logits


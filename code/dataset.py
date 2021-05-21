import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Tuple


def make_datasets(path: str):
    # split out one `context` totally, instead of random split.
    raw_data = torch.Tensor(np.load(path))
    cls_ctx_mat = torch.zeros((10, 10))
    for i, feature in enumerate(raw_data):
        cls_ctx_mat[int(feature[-1]), int(feature[-2])] += 1
    nonzero_index = cls_ctx_mat.nonzero(as_tuple=False).reshape((10, 7, 2))
    random_index = (torch.rand((10,)) * 7).floor().long()
    train_set, eval_set = [], []
    for i, feature in enumerate(raw_data):
        cls_idx, ctx_idx = int(feature[-1]), int(feature[-2])
        if nonzero_index[cls_idx, random_index[cls_idx], -1] == ctx_idx:
            eval_set.append(feature)
        else:
            train_set.append(feature)
    # split train_set into different 10 classes
    train_class_sets = [[] for _ in range(10)]
    for feature in train_set:
        cls_idx = int(feature[-1])
        train_class_sets[cls_idx].append(feature)
    train_set = torch.stack(train_set, dim=0)
    eval_set = torch.stack(eval_set, dim=0)
    train_class_datasets = [
        NicoNaiveTrainDataset(torch.stack(_dataset, dim=0)) for _dataset in train_class_sets
    ]
    return (
        NicoNaiveTrainDataset(train_set),
        NicoNaiveTrainDataset(eval_set),
        train_class_datasets
    )


class NicoNaiveTrainDataset(Dataset):

    def __init__(self, data_array: torch.Tensor):
        self.raw_data = data_array
        self.features = self.raw_data[:, :512].type(torch.float32)
        self.ctx_label = self.raw_data[:, -2].type(torch.int64)
        self.cls_label = self.raw_data[:, -1].type(torch.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.cls_label[index], self.ctx_label[index]


class NicoNaiveTestDataset(Dataset):

    def __init__(self, path: str):
        self.path = path
        self.raw_data = torch.Tensor(np.load(path))
        self.features = self.raw_data[:, :512].type(torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

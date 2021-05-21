import torch

def onehot(label: torch.Tensor, N: int):
    B = label.shape[0]
    return torch.zeros(B, N).to(label).scatter_(1, label.unsqueeze(dim=1), 1)

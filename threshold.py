import torch

def soft_threshold(x, T):
    return torch.copysign(torch.abs(x)-T, x)

def hard_threshold(x, T):
    return torch.where(torch.abs(x)>=T, x, 0)
    
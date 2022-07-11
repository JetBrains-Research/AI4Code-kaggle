import torch


def extract_value(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return x

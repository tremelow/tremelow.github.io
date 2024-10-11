import torch

DEFAULT_SEED = sum(ord(c) ** 2 for c in "R5.A.12-ModMath")


def torch_rng(seed=DEFAULT_SEED):
    return torch.Generator().manual_seed(seed)

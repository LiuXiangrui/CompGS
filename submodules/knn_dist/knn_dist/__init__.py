import torch
from ._C import distCUDA2


def distance(*args, **kwargs):
    return distCUDA2(*args, **kwargs)
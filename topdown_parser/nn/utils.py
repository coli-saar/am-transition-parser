import torch
from allennlp.nn.util import get_device_of


def get_device_id(t : torch.Tensor):
    d = get_device_of(t)
    if d < 0:
        return None
    return d

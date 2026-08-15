
import torch
import torch.nn as nn

from annotator.uniformer.mmcv import build_from_cfg
from .registry import DROPOUT_LAYERS

def drop_path(x, drop_prob=0., training=False):
    
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output

@DROPOUT_LAYERS.register_module()
class DropPath(nn.Module):
    
    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

@DROPOUT_LAYERS.register_module()
class Dropout(nn.Dropout):
    
    def __init__(self, drop_prob=0.5, inplace=False):
        super().__init__(p=drop_prob, inplace=inplace)

def build_dropout(cfg, default_args=None):
    
    return build_from_cfg(cfg, DROPOUT_LAYERS, default_args)

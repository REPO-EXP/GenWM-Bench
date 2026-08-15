
import torch
import torch.nn as nn
import torch.nn.functional as F

from annotator.uniformer.mmcv.utils import TORCH_VERSION, build_from_cfg, digit_version
from .registry import ACTIVATION_LAYERS

for module in [
        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ReLU6, nn.ELU,
        nn.Sigmoid, nn.Tanh
]:
    ACTIVATION_LAYERS.register_module(module=module)

@ACTIVATION_LAYERS.register_module(name='Clip')
@ACTIVATION_LAYERS.register_module()
class Clamp(nn.Module):
    
    def __init__(self, min=-1., max=1.):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        
        return torch.clamp(x, min=self.min, max=self.max)

class GELU(nn.Module):
    
    def forward(self, input):
        return F.gelu(input)

if (TORCH_VERSION == 'parrots'
        or digit_version(TORCH_VERSION) < digit_version('1.4')):
    ACTIVATION_LAYERS.register_module(module=GELU)
else:
    ACTIVATION_LAYERS.register_module(module=nn.GELU)

def build_activation_layer(cfg):
    
    return build_from_cfg(cfg, ACTIVATION_LAYERS)

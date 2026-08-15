
import torch.nn as nn

from .registry import ACTIVATION_LAYERS

@ACTIVATION_LAYERS.register_module()
class HSigmoid(nn.Module):
    
    def __init__(self, bias=1.0, divisor=2.0, min_value=0.0, max_value=1.0):
        super(HSigmoid, self).__init__()
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = (x + self.bias) / self.divisor

        return x.clamp_(self.min_value, self.max_value)

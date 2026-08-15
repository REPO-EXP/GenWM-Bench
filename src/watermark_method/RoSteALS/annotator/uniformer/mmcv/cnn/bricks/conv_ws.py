
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import CONV_LAYERS

def conv_ws_2d(input,
               weight,
               bias=None,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               eps=1e-5):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

@CONV_LAYERS.register_module('ConvWS')
class ConvWS2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5):
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups, self.eps)

@CONV_LAYERS.register_module(name='ConvAWS')
class ConvAWS2d(nn.Conv2d):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.register_buffer('weight_gamma',
                             torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('weight_beta',
                             torch.zeros(self.out_channels, 1, 1, 1))

    def _get_weight(self, weight):
        weight_flat = weight.view(weight.size(0), -1)
        mean = weight_flat.mean(dim=1).view(-1, 1, 1, 1)
        std = torch.sqrt(weight_flat.var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        weight = (weight - mean) / std
        weight = self.weight_gamma * weight + self.weight_beta
        return weight

    def forward(self, x):
        weight = self._get_weight(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        
        self.weight_gamma.data.fill_(-1)
        local_missing_keys = []
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, local_missing_keys,
                                      unexpected_keys, error_msgs)
        if self.weight_gamma.data.mean() > 0:
            for k in local_missing_keys:
                missing_keys.append(k)
            return
        weight = self.weight.data
        weight_flat = weight.view(weight.size(0), -1)
        mean = weight_flat.mean(dim=1).view(-1, 1, 1, 1)
        std = torch.sqrt(weight_flat.var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_beta.data.copy_(mean)
        self.weight_gamma.data.copy_(std)
        missing_gamma_beta = [
            k for k in local_missing_keys
            if k.endswith('weight_gamma') or k.endswith('weight_beta')
        ]
        for k in missing_gamma_beta:
            local_missing_keys.remove(k)
        for k in local_missing_keys:
            missing_keys.append(k)

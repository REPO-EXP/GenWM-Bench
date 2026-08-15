
import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _triple

from .registry import CONV_LAYERS, UPSAMPLE_LAYERS

if torch.__version__ == 'parrots':
    TORCH_VERSION = torch.__version__
else:
    
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])

def obsolete_torch_version(torch_version, version_threshold):
    return torch_version == 'parrots' or torch_version <= version_threshold

class NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None

@CONV_LAYERS.register_module('Conv', force=True)
class Conv2d(nn.Conv2d):

    def forward(self, x):
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d in zip(x.shape[-2:], self.kernel_size,
                                     self.padding, self.stride, self.dilation):
                o = (i + 2 * p - (d * (k - 1) + 1)) // s + 1
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)

@CONV_LAYERS.register_module('Conv3d', force=True)
class Conv3d(nn.Conv3d):

    def forward(self, x):
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d in zip(x.shape[-3:], self.kernel_size,
                                     self.padding, self.stride, self.dilation):
                o = (i + 2 * p - (d * (k - 1) + 1)) // s + 1
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)

@CONV_LAYERS.register_module()
@CONV_LAYERS.register_module('deconv')
@UPSAMPLE_LAYERS.register_module('deconv', force=True)
class ConvTranspose2d(nn.ConvTranspose2d):

    def forward(self, x):
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(x.shape[-2:], self.kernel_size,
                                         self.padding, self.stride,
                                         self.dilation, self.output_padding):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)

@CONV_LAYERS.register_module()
@CONV_LAYERS.register_module('deconv3d')
@UPSAMPLE_LAYERS.register_module('deconv3d', force=True)
class ConvTranspose3d(nn.ConvTranspose3d):

    def forward(self, x):
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(x.shape[-3:], self.kernel_size,
                                         self.padding, self.stride,
                                         self.dilation, self.output_padding):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)

class MaxPool2d(nn.MaxPool2d):

    def forward(self, x):
        
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 9)):
            out_shape = list(x.shape[:2])
            for i, k, p, s, d in zip(x.shape[-2:], _pair(self.kernel_size),
                                     _pair(self.padding), _pair(self.stride),
                                     _pair(self.dilation)):
                o = (i + 2 * p - (d * (k - 1) + 1)) / s + 1
                o = math.ceil(o) if self.ceil_mode else math.floor(o)
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            return empty

        return super().forward(x)

class MaxPool3d(nn.MaxPool3d):

    def forward(self, x):
        
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 9)):
            out_shape = list(x.shape[:2])
            for i, k, p, s, d in zip(x.shape[-3:], _triple(self.kernel_size),
                                     _triple(self.padding),
                                     _triple(self.stride),
                                     _triple(self.dilation)):
                o = (i + 2 * p - (d * (k - 1) + 1)) / s + 1
                o = math.ceil(o) if self.ceil_mode else math.floor(o)
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            return empty

        return super().forward(x)

class Linear(torch.nn.Linear):

    def forward(self, x):
        
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 5)):
            out_shape = [x.shape[0], self.out_features]
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)

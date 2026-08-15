
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['fused_bias_leakyrelu'])

class FusedBiasLeakyReLUFunctionBackward(Function):
    
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = ext_module.fused_bias_leakyrelu(
            grad_output,
            empty,
            out,
            act=3,
            grad=1,
            alpha=negative_slope,
            scale=scale)

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors

        gradgrad_out = ext_module.fused_bias_leakyrelu(
            gradgrad_input,
            gradgrad_bias.to(out.dtype),
            out,
            act=3,
            grad=1,
            alpha=ctx.negative_slope,
            scale=ctx.scale)

        return gradgrad_out, None, None, None

class FusedBiasLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)

        out = ext_module.fused_bias_leakyrelu(
            input,
            bias,
            empty,
            act=3,
            grad=0,
            alpha=negative_slope,
            scale=scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedBiasLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale)

        return grad_input, grad_bias, None, None

class FusedBiasLeakyReLU(nn.Module):
    
    def __init__(self, num_channels, negative_slope=0.2, scale=2**0.5):
        super(FusedBiasLeakyReLU, self).__init__()

        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_bias_leakyrelu(input, self.bias, self.negative_slope,
                                    self.scale)

def fused_bias_leakyrelu(input, bias, negative_slope=0.2, scale=2**0.5):
    
    if not input.is_cuda:
        return bias_leakyrelu_ref(input, bias, negative_slope, scale)

    return FusedBiasLeakyReLUFunction.apply(input, bias.to(input.dtype),
                                            negative_slope, scale)

def bias_leakyrelu_ref(x, bias, negative_slope=0.2, scale=2**0.5):

    if bias is not None:
        assert bias.ndim == 1
        assert bias.shape[0] == x.shape[1]
        x = x + bias.reshape([-1 if i == 1 else 1 for i in range(x.ndim)])

    x = F.leaky_relu(x, negative_slope)
    if scale != 1:
        x = x * scale

    return x

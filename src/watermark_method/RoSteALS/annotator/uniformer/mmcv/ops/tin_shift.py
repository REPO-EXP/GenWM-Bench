
import torch
import torch.nn as nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext',
                                 ['tin_shift_forward', 'tin_shift_backward'])

class TINShiftFunction(Function):

    @staticmethod
    def forward(ctx, input, shift):
        C = input.size(2)
        num_segments = shift.size(1)
        if C // num_segments <= 0 or C % num_segments != 0:
            raise ValueError('C should be a multiple of num_segments, '
                             )

        ctx.save_for_backward(shift)

        out = torch.zeros_like(input)
        ext_module.tin_shift_forward(input, shift, out)

        return out

    @staticmethod
    def backward(ctx, grad_output):

        shift = ctx.saved_tensors[0]
        data_grad_input = grad_output.new(*grad_output.size()).zero_()
        shift_grad_input = shift.new(*shift.size()).zero_()
        ext_module.tin_shift_backward(grad_output, shift, data_grad_input)

        return data_grad_input, shift_grad_input

tin_shift = TINShiftFunction.apply

class TINShift(nn.Module):
    
    def forward(self, input, shift):
        
        return tin_shift(input, shift)

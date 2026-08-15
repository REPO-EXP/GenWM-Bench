
import functools
import warnings
from collections import abc
from inspect import getfullargspec

import numpy as np
import torch
import torch.nn as nn

from annotator.uniformer.mmcv.utils import TORCH_VERSION, digit_version
from .dist_utils import allreduce_grads as _allreduce_grads

try:
    
    from torch.cuda.amp import autocast
except ImportError:
    pass

def cast_tensor_type(inputs, src_type, dst_type):
    
    if isinstance(inputs, nn.Module):
        return inputs
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs

def auto_fp16(apply_to=None, out_fp32=False):
    
    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@auto_fp16 can only be used to decorate the '
                                )
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)

            args_info = getfullargspec(old_func)
            
            args_to_cast = args_info.args if apply_to is None else apply_to
            
            new_args = []
            
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value
            
            if (TORCH_VERSION != 'parrots' and
                    digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
                with autocast(enabled=True):
                    output = old_func(*new_args, **new_kwargs)
            else:
                output = old_func(*new_args, **new_kwargs)
            
            if out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper

def force_fp32(apply_to=None, out_fp16=False):
    
    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the '
                                )
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            
            args_info = getfullargspec(old_func)
            
            args_to_cast = args_info.args if apply_to is None else apply_to
            
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            
            if (TORCH_VERSION != 'parrots' and
                    digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
                with autocast(enabled=False):
                    output = old_func(*new_args, **new_kwargs)
            else:
                output = old_func(*new_args, **new_kwargs)
            
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper

def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    warnings.warning(
        
        )
    _allreduce_grads(params, coalesce=coalesce, bucket_size_mb=bucket_size_mb)

def wrap_fp16_model(model):
    
    if (TORCH_VERSION == 'parrots'
            or digit_version(TORCH_VERSION) < digit_version('1.6.0')):
        
        model.half()
        
        patch_norm_fp32(model)
    
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True

def patch_norm_fp32(module):
    
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
        module.float()
        if isinstance(module, nn.GroupNorm) or torch.__version__ < '1.3':
            module.forward = patch_forward_method(module.forward, torch.half,
                                                  torch.float)
    for child in module.children():
        patch_norm_fp32(child)
    return module

def patch_forward_method(func, src_type, dst_type, convert_output=True):
    
    def new_forward(*args, **kwargs):
        output = func(*cast_tensor_type(args, src_type, dst_type),
                      **cast_tensor_type(kwargs, src_type, dst_type))
        if convert_output:
            output = cast_tensor_type(output, dst_type, src_type)
        return output

    return new_forward

class LossScaler:
    
    def __init__(self,
                 init_scale=2**32,
                 mode='dynamic',
                 scale_factor=2.,
                 scale_window=1000):
        self.cur_scale = init_scale
        self.cur_iter = 0
        assert mode in ('dynamic',
                        ), 'mode can only be dynamic or static'
        self.mode = mode
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    def has_overflow(self, params):
        
        if self.mode != 'dynamic':
            return False
        for p in params:
            if p.grad is not None and LossScaler._has_inf_or_nan(p.grad.data):
                return True
        return False

    def _has_inf_or_nan(x):
        
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if 'value cannot be converted' not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf')                    or cpu_sum != cpu_sum:
                return True
            return False

    def update_scale(self, overflow):
        
        if self.mode != 'dynamic':
            return
        if overflow:
            self.cur_scale = max(self.cur_scale / self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) %                    self.scale_window == 0:
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    def state_dict(self):
        
        return dict(
            cur_scale=self.cur_scale,
            cur_iter=self.cur_iter,
            mode=self.mode,
            last_overflow_iter=self.last_overflow_iter,
            scale_factor=self.scale_factor,
            scale_window=self.scale_window)

    def load_state_dict(self, state_dict):
        
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        self.mode = state_dict['mode']
        self.last_overflow_iter = state_dict['last_overflow_iter']
        self.scale_factor = state_dict['scale_factor']
        self.scale_window = state_dict['scale_window']

    @property
    def loss_scale(self):
        return self.cur_scale

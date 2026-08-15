
import sys
from collections.abc import Iterable
from runpy import run_path
from shlex import split
from typing import Any, Dict, List
from unittest.mock import patch

def check_python_script(cmd):
    
    args = split(cmd)
    if args[0] == 'python':
        args = args[1:]
    with patch.object(sys, 'argv', args):
        run_path(args[0], run_name='__main__')

def _any(judge_result):
    
    if not isinstance(judge_result, Iterable):
        return judge_result

    try:
        for element in judge_result:
            if _any(element):
                return True
    except TypeError:
        
        if judge_result:
            return True
    return False

def assert_dict_contains_subset(dict_obj: Dict[Any, Any],
                                expected_subset: Dict[Any, Any]) -> bool:
    
    for key, value in expected_subset.items():
        if key not in dict_obj.keys() or _any(dict_obj[key] != value):
            return False
    return True

def assert_attrs_equal(obj: Any, expected_attrs: Dict[str, Any]) -> bool:
    
    for attr, value in expected_attrs.items():
        if not hasattr(obj, attr) or _any(getattr(obj, attr) != value):
            return False
    return True

def assert_dict_has_keys(obj: Dict[str, Any],
                         expected_keys: List[str]) -> bool:
    
    return set(expected_keys).issubset(set(obj.keys()))

def assert_keys_equal(result_keys: List[str], target_keys: List[str]) -> bool:
    
    return set(result_keys) == set(target_keys)

def assert_is_norm_layer(module) -> bool:
    
    from .parrots_wrapper import _BatchNorm, _InstanceNorm
    from torch.nn import GroupNorm, LayerNorm
    norm_layer_candidates = (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm)
    return isinstance(module, norm_layer_candidates)

def assert_params_all_zeros(module) -> bool:
    
    weight_data = module.weight.data
    is_weight_zero = weight_data.allclose(
        weight_data.new_zeros(weight_data.size()))

    if hasattr(module, 'bias') and module.bias is not None:
        bias_data = module.bias.data
        is_bias_zero = bias_data.allclose(
            bias_data.new_zeros(bias_data.size()))
    else:
        is_bias_zero = True

    return is_weight_zero and is_bias_zero

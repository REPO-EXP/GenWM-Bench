
import collections.abc
import functools
import itertools
import subprocess
import warnings
from collections import abc
from importlib import import_module
from inspect import getfullargspec
from itertools import repeat

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def is_str(x):
    
    return isinstance(x, str)

def import_modules_from_strings(imports, allow_failed_imports=False):
    
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            )
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                )
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported

def iter_cast(inputs, dst_type, return_type=None):
    
    if not isinstance(inputs, abc.Iterable):
        raise TypeError('inputs must be an iterable object')
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')

    out_iterable = map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)

def list_cast(inputs, dst_type):
    
    return iter_cast(inputs, dst_type, return_type=list)

def tuple_cast(inputs, dst_type):
    
    return iter_cast(inputs, dst_type, return_type=tuple)

def is_seq_of(seq, expected_type, seq_type=None):
    
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def is_list_of(seq, expected_type):
    
    return is_seq_of(seq, expected_type, seq_type=list)

def is_tuple_of(seq, expected_type):
    
    return is_seq_of(seq, expected_type, seq_type=tuple)

def slice_list(in_list, lens):
    
    if isinstance(lens, int):
        assert len(in_list) % lens == 0
        lens = [lens] * int(len(in_list) / lens)
    if not isinstance(lens, list):
        raise TypeError('"indices" must be an integer or a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError('sum of lens and list length does not '
                         )
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list

def concat_list(in_list):
    
    return list(itertools.chain(*in_list))

def check_prerequisites(
        prerequisites,
        checker,
        msg_tmpl='Prerequisites "{}" are required in method "{}" but not '
        ):  
    
    def wrap(func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = [prerequisites] if isinstance(
                prerequisites, str) else prerequisites
            missing = []
            for item in requirements:
                if not checker(item):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(', '.join(missing), func.__name__))
                raise RuntimeError('Prerequisites not meet.')
            else:
                return func(*args, **kwargs)

        return wrapped_func

    return wrap

def _check_py_package(package):
    try:
        import_module(package)
    except ImportError:
        return False
    else:
        return True

def _check_executable(cmd):
    if subprocess.call(f'which {cmd}', shell=True) != 0:
        return False
    else:
        return True

def requires_package(prerequisites):
    
    return check_prerequisites(prerequisites, checker=_check_py_package)

def requires_executable(prerequisites):
    
    return check_prerequisites(prerequisites, checker=_check_executable)

def deprecated_api_warning(name_dict, cls_name=None):
    
    def api_warning_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            
            args_info = getfullargspec(old_func)
            
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f'{cls_name}.{func_name}'
            if args:
                arg_names = args_info.args[:len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            
                            )
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:

                        assert dst_arg_name not in kwargs, (
                            
                            )

                        warnings.warn(
                            
                            )
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper

def is_method_overridden(method, base_class, derived_class):
    
    assert isinstance(base_class, type),        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method

def has_method(obj: object, method: str) -> bool:
    
    return hasattr(obj, method) and callable(getattr(obj, method))

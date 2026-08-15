
from .registry import MODULE_WRAPPERS

def is_module_wrapper(module):
    
    module_wrappers = tuple(MODULE_WRAPPERS.module_dict.values())
    return isinstance(module, module_wrappers)

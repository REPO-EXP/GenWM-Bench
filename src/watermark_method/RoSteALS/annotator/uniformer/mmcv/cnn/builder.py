
from ..runner import Sequential
from ..utils import Registry, build_from_cfg

def build_model_from_cfg(cfg, registry, default_args=None):
    
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

MODELS = Registry('model', build_func=build_model_from_cfg)

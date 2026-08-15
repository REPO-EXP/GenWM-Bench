import inspect
import platform

from .registry import PLUGIN_LAYERS

if platform.system() == 'Windows':
    import regex as re
else:
    import re

def infer_abbr(class_type):
    
    def camel2snack(word):
        
        word = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', word)
        word = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', word)
        word = word.replace('-', '_')
        return word.lower()

    if not inspect.isclass(class_type):
        raise TypeError(
            )
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    else:
        return camel2snack(class_type.__name__)

def build_plugin_layer(cfg, postfix='', **kwargs):
    
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in PLUGIN_LAYERS:
        raise KeyError(f'Unrecognized plugin type {layer_type}')

    plugin_layer = PLUGIN_LAYERS.get(layer_type)
    abbr = infer_abbr(plugin_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    layer = plugin_layer(**kwargs, **cfg_)

    return name, layer

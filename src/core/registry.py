
class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, name=None):
        def _register(cls):
            key = name if name else cls.__name__
            self._module_dict[key] = cls
            return cls
        return _register

    def build(self, config):
        
        if config is None: return None
        if 'name' not in config:
            raise ValueError(f"Config for {self._name} must contain 'name'")
        
        target_name = config['name']
        if target_name not in self._module_dict:
            raise KeyError(f"{target_name} not found in {self._name}")
            
        cls = self._module_dict[target_name]
        params = {k: v for k, v in config.items() if k != 'name'}
        return cls(**params)

MODELS = Registry("Models")
WATERMARKS = Registry("Watermarks")
ATTACKS = Registry("Attacks")
DATASETS = Registry("Datasets")
METRICS = Registry("Metrics")
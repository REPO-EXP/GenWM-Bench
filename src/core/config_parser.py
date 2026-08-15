
import yaml
import os

class ConfigManager:
    @staticmethod
    def load_yaml(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def deep_update(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base:
                ConfigManager.deep_update(base[k], v)
            else:
                base[k] = v
        return base

    @classmethod
    def load_experiment_config(cls, path):
        exp_config = cls.load_yaml(path)
        final_config = {}

        if 'defaults' in exp_config:
            defaults = exp_config.pop('defaults')
            for key, file_path in defaults.items():
                base_cfg = cls.load_yaml(file_path)
                final_config[key] = base_cfg
        
        cls.deep_update(final_config, exp_config)
        return final_config
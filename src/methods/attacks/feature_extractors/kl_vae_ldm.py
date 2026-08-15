import torch
import sys
import os
from .base_encoder import BaseEncoder
from omegaconf import OmegaConf
import importlib

def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class KLVAEEmbedding(BaseEncoder):
    def __init__(self, model_name):
        super().__init__()
        
        model_root = os.path.abspath(f"./data/models/{model_name}")
        
        if os.path.exists(model_root):
            if model_root not in sys.path:
                print(f"[KLVAE] Injecting model root to sys.path: {model_root}")
                sys.path.insert(0, model_root)
        else:
            print(f"[KLVAE Warning] Model directory not found at: {model_root}")

        self.model = self.get_model(model_name)

    def load_model_from_config(self, config, ckpt):
        print(f"Loading model from {ckpt}")
        
        pl_sd = torch.load(
            ckpt,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=False  
        )
        
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        
        if len(m) > 0:
            print("missing keys:")
            print(m)
        if len(u) > 0:
            print("unexpected keys:")
            print(u)
            
        model.eval()
        return model

    def get_model(self, name):
        
        base_path = f"./data/models/{name}"
        config_path = os.path.join(base_path, "config.yaml")
        model_path = os.path.join(base_path, "model.ckpt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print(f"[KLVAE] Config: {config_path}")
        config = OmegaConf.load(config_path)
        model = self.load_model_from_config(config, model_path)
        return model

    def forward(self, images):
        
        images = 2.0 * images - 1.0
        
        output = self.model.encode(images)
        if hasattr(output, 'mode'):
            z = output.mode()
        else:
            z = output.sample()
            
        return z
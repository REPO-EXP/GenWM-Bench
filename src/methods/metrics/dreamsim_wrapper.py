import os
import sys
import torch
import torch.nn.functional as F
import importlib
import contextlib
from src.core.interfaces import BaseMetric
from src.core.registry import METRICS

@contextlib.contextmanager
def clean_sys_path():
    
    original_path = list(sys.path)
    CONFLICT_SOURCE_DIR = "watermark_method"
    
    clean_path_list = []
    removed_count = 0
    
    for p in original_path:
        if CONFLICT_SOURCE_DIR in str(p):
            removed_count += 1
            continue
        clean_path_list.append(p)
            
    sys.path[:] = clean_path_list
    
    conflict_module_names = ['utils']
    saved_modules = {}
    
    for mod_name in conflict_module_names:
        if mod_name in sys.modules:
            mod_path = getattr(sys.modules[mod_name], '__file__', '')
            if mod_path and CONFLICT_SOURCE_DIR in str(mod_path):
                saved_modules[mod_name] = sys.modules.pop(mod_name)
    
    if removed_count > 0 and not hasattr(sys, "_dreamsim_log_once"):
        print(f"[DreamSim Fix] Temporarily masked {removed_count} path(s) and cleared {list(saved_modules.keys())} from modules.")
        setattr(sys, "_dreamsim_log_once", True)
    
    try:
        yield
    finally:
        sys.path[:] = original_path
        
        for mod_name in conflict_module_names:
            if mod_name in sys.modules and mod_name in saved_modules:
                sys.modules.pop(mod_name)
            if mod_name in saved_modules:
                sys.modules[mod_name] = saved_modules[mod_name]

@METRICS.register("DreamSimMetric")
class DreamSimMetric(BaseMetric):
    def __init__(self, config, device="cuda", **kwargs):
        super().__init__(config=config, device=device)
        self.device = device
        
        self.model_type = self.config.get('model_type', 'ensemble')
        self.cache_dir = self.config.get('cache_dir', './data/models/dreamsim_models/')
        
        self.model = None
        self.preprocess = None
        
        try:
            
            with clean_sys_path():
                self.dreamsim_func = self._import_dreamsim_func()
            
            if self.dreamsim_func:
                print(f"[Metrics] Loading DreamSim ({self.model_type}) on {self.device}...")
                
                with clean_sys_path():
                    self.model, self.preprocess = self.dreamsim_func(
                        pretrained=True,
                        device=self.device,
                        cache_dir=self.cache_dir,
                        dreamsim_type=self.model_type
                    )
                self.model.eval()
                
        except Exception as e:
            print(f"[Error] Failed to initialize DreamSim: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def _import_dreamsim_func(self):
        
        try:
            import dreamsim
            return dreamsim.dreamsim
        except ImportError:
            pass
            
        try:
            import src.eval_metrics.dreamsim.model as local_dreamsim
            if hasattr(local_dreamsim, 'dreamsim'):
                return local_dreamsim.dreamsim
        except ImportError:
            pass
            
        print("[Warning] DreamSim package/module not found.")
        return None

    def _prepare_image(self, img):
        if img is None: return None
        
        if not torch.is_tensor(img):
            if self.preprocess:
                return self.preprocess(img).to(self.device)
            return None

        img = img.to(self.device)
        if img.ndim == 3: img = img.unsqueeze(0)
            
        if img.shape[-2:] != (224, 224):
            img = F.interpolate(img, size=(224, 224), mode='bicubic', align_corners=False)
        
        if img.min() < 0: 
            img = (img + 1) / 2.0
            
        return img

    def calculate(self, img_gen_wm=None, img_gen_clean=None, **kwargs) -> dict:
        if self.model is None: return {"dreamsim": -1.0}

        input_img = img_gen_wm if img_gen_wm is not None else img_gen_clean
        target_img = kwargs.get('img_gt')
        if target_img is None: target_img = img_gen_clean

        if input_img is None or target_img is None: return {}

        try:
            t_input = self._prepare_image(input_img)
            t_target = self._prepare_image(target_img)

            with torch.no_grad():
                dist = self.model(t_input, t_target)
            return {"dreamsim": float(dist.item())}
        except Exception as e:
            print(f"[Error] DreamSim calc failed: {e}")
            return {"dreamsim": -1.0}
    
    def compute_aggregate(self, context: dict) -> dict:
        return {}
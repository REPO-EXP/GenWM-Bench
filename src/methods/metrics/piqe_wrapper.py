import numpy as np
import torch
from src.core.interfaces import BaseMetric
from src.core.registry import METRICS

try:
    import pypiqe
except ImportError:
    pypiqe = None

@METRICS.register("PIQEMetric")
class PIQEMetric(BaseMetric):
    def __init__(self, config=None, **kwargs):
        if config is None: config = {}
        config.update(kwargs)
        super().__init__(config, **kwargs)
        
        if pypiqe is None:
            print("🔴 [Metrics] Warning: 'pypiqe' library not found. PIQE will be skipped.")
            self.ready = False
        else:
            self.ready = True

    def calculate(self, **kwargs) -> dict:
        if not self.ready: return {}
        
        img = kwargs.get('img_gen_wm')
        if img is None: return {}
        
        try:
            
            img_np = self._preprocess_input(img)
            
            score, _, _, _ = pypiqe.piqe(img_np)
            
            return {"piqe": float(score)}
            
        except Exception as e:
            
            print(f"[Metrics] PIQE calc error: {e}")
            return {}

    def _preprocess_input(self, img):
        
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 4:
                img = img.squeeze(0)
            
            if img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0)
            
            img = img.numpy()
            
        if isinstance(img, np.ndarray):
            
            if img.dtype != np.uint8:
                if img.max() <= 1.05:
                    img = (img * 255.0)
                img = img.clip(0, 255).astype(np.uint8)

        img = np.ascontiguousarray(img)
        
        return img

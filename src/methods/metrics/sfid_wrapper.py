import torch
from torchvision.transforms.functional import to_tensor
from src.core.interfaces import BaseMetric
from src.core.registry import METRICS

try:
    from src.eval_metrics.sifid.sifid import SIFID
except ImportError:
    SIFID = None
    print("[Warning] SIFID module not found in src.eval_metrics.sifid.sifid")

@METRICS.register("SFIDMetric")
class SFIDMetric(BaseMetric):
    def __init__(self, config=None, **kwargs):
        
        if config is None: config = {}
        config.update(kwargs)
        super().__init__(config)
        
        self.sifid_model = None
        if SIFID is not None:
            
            dims = config.get('dims', 64)
            print(f"[Metrics] Loading SIFID model (dims={dims}) on {self.device}...")
            
            try:
                
                self.sifid_model = SIFID(dims=dims, device=self.device)
            except Exception as e:
                print(f"[Metrics] Failed to init SIFID: {e}")

    def calculate(self, **kwargs) -> dict:
        
        if self.sifid_model is None:
            return {}
        
        img_ref = kwargs.get('img_gen_clean') 
        img_dist = kwargs.get('img_gen_wm')   

        if img_ref is None or img_dist is None:
            return {}

        try:
            
            t_ref = self._preprocess(img_ref)
            t_dist = self._preprocess(img_dist)
            
            with torch.no_grad():
                score = self.sifid_model(t_ref, t_dist)
            
            return {"sfid": float(score)}
            
        except Exception as e:
            print(f"[Metrics] SFID calculation failed: {e}")
            return {}

    def _preprocess(self, img):
        
        if not isinstance(img, torch.Tensor):
            
            img = to_tensor(img)
        
        img = img.to(self.device)
        
        if img.ndim == 4:
            img = img.squeeze(0)
            
        return img * 2.0 - 1.0

    def compute_aggregate(self, context) -> dict:
        
        return {}
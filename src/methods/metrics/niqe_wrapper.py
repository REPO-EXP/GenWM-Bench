import torch
from src.core.interfaces import BaseMetric
from src.core.registry import METRICS
from torchvision.transforms.functional import to_tensor

try:
    import pyiqa
except ImportError:
    pyiqa = None
    print("[Warning] pyiqa not installed. NIQE will be 0.")

@METRICS.register("NIQEMetric")
class NIQEMetric(BaseMetric):
    def __init__(self, config=None, **kwargs):
        if config is None: config = {}
        config.update(kwargs)
        super().__init__(config)
        
        self.metric = None
        if pyiqa is not None:
            print(f"[Metrics] Loading NIQE metric on {self.device}...")
            self.metric = pyiqa.create_metric('niqe', device=self.device)

    def calculate(self, **kwargs) -> dict:
        if self.metric is None: return {}
        img_wm = kwargs.get('img_gen_wm')
        if img_wm is None: return {}
        
        try:
            if not isinstance(img_wm, torch.Tensor):
                img_tensor = to_tensor(img_wm).unsqueeze(0).to(self.device)
            else:
                img_tensor = img_wm.to(self.device)
                if img_tensor.ndim == 3: img_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():
                score = self.metric(img_tensor)
            return {"niqe": score.item()}
        except Exception as e:
            print(f"[Metrics] NIQE error: {e}")
            return {}
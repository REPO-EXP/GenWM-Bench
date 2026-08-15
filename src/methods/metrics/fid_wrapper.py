import torch
import os
from src.core.interfaces import BaseMetric
from src.core.registry import METRICS

try:
    from src.eval_metrics.pytorch_fid.fid_score import calculate_fid_given_paths
except ImportError:
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError:
        calculate_fid_given_paths = None
        print("[Warning] pytorch-fid not found. FID will be 0.")

@METRICS.register("FIDMetric")
class FIDMetric(BaseMetric):
    def __init__(self, config, device="cuda", **kwargs):
        super().__init__(config=config, device=device)
        self.device = device
        
        self.batch_size = self.config.get('batch_size', 50)
        self.dims = self.config.get('dims', 2048)
        self.num_workers = self.config.get('num_workers', 4)

    def calculate(self, **kwargs) -> dict:
        
        return {}

    def compute_aggregate(self, context: dict) -> dict:
        
        if calculate_fid_given_paths is None:
            return {"fid": -1.0}

        path_ref = context.get('path_ref')
        path_pred = context.get('path_pred')

        if not path_ref or not path_pred:
            
            return {}

        if not os.path.exists(path_ref) or not os.path.exists(path_pred):
            print(f"[FID Error] Paths do not exist:\n  Ref: {path_ref}\n  Pred: {path_pred}")
            return {"fid": -1.0}

        try:
            
            print(f"[FID] Calculating: {os.path.basename(path_pred)} vs {os.path.basename(path_ref)}")
            
            fid_value = calculate_fid_given_paths(
                paths=[path_ref, path_pred],
                batch_size=self.batch_size,
                device=self.device,
                dims=self.dims,
                num_workers=self.num_workers
            )
            result = {"fid": float(fid_value)}
            print(f"   -> [FID Result] {result}")
            return result
            
        except Exception as e:
            print(f"[FID Error] Calculation failed: {e}")
            return {"fid": -1.0}
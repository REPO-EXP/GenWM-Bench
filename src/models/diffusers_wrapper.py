import torch
from src.core.registry import MODELS
from src.core.paths import resolve_model_path

@MODELS.register("StableDiffusion")
class StableDiffusionWrapper:
    def __init__(self, model_id, device='cuda', dtype='fp16', **kwargs):
        
        from diffusers import StableDiffusionPipeline

        model_id = resolve_model_path(model_id)
        print(f"   [Model] Loading Diffusers: {model_id} ({dtype})...")

        kwargs.pop('dtype', None) 
        kwargs.pop('torch_dtype', None)
        
        if dtype == 'fp16':
            internal_torch_dtype = torch.float16
        else:
            internal_torch_dtype = torch.float32
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                
                torch_dtype=internal_torch_dtype, 
                safety_checker=None, 
                requires_safety_checker=False,
                
                **kwargs 
            )
        except Exception as e:
            print(f"   [Warning] Load failed: {e}")
            raise e

        self.device = device
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def get_pipe(self):
        return self.pipe
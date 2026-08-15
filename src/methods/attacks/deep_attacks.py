import torch
from PIL import Image
from torchvision import transforms
from src.core.interfaces import BaseAttack
from src.core.registry import ATTACKS

@ATTACKS.register("VAECompression")
class VAECompression(BaseAttack):
    def __init__(self, model_name='bmshj2018-factorized', quality=1, metric='mse', device='cuda'):
        self.model_name = model_name
        self.quality = int(quality)
        self.device = device
        self.model = self._load_model(model_name, self.quality, metric)

    def _load_model(self, model_name, quality, metric):
        try:
            from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor
        except ImportError:
            raise ImportError("Please install compressai: pip install compressai")

        models = {
            'bmshj2018-factorized': bmshj2018_factorized,
            'bmshj2018-hyperprior': bmshj2018_hyperprior,
            'mbt2018-mean': mbt2018_mean,
            'mbt2018': mbt2018,
            'cheng2020-anchor': cheng2020_anchor
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown VAE model: {model_name}")
            
        print(f"   [VAE] Loading {model_name} (quality={quality})...")
        net = models[model_name](quality=quality, pretrained=True).eval().to(self.device)
        return net

    def apply(self, image: Image.Image) -> Image.Image:
        
        w, h = image.size
        img_resized = image.resize((512, 512))
        
        x = transforms.ToTensor()(img_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(x)
            x_hat = out['x_hat'].clamp(0, 1)
        
        rec = transforms.ToPILImage()(x_hat.squeeze().cpu())
        return rec.resize((w, h))

    def get_param_str(self):
        return f"VAE_q{self.quality}"

import torch
from PIL import Image

from src.core.paths import resolve_model_path

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    PNDMScheduler, DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler, LMSDiscreteScheduler, UniPCMultistepScheduler,
    HeunDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler
)

SCHEDULER_MAP = {
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
    "Euler": EulerDiscreteScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DPM++ 2M SDE": DPMSolverMultistepScheduler,
    "LMS": LMSDiscreteScheduler,
    "UniPC": UniPCMultistepScheduler,
    "Heun": HeunDiscreteScheduler,
    "DPM2": KDPM2DiscreteScheduler,
    "DPM2 a": KDPM2AncestralDiscreteScheduler,
}

@ATTACKS.register("DiffusionRegeneration")
class DiffusionRegeneration(BaseAttack):
    def __init__(self, 
                
                 model_id="../../model/stable-diffusion-v1-5", 
                 steps=20, 
                 strength=0.4, 
                 scheduler="PNDM", 
                 iterations=1, 
                 device='cuda'):
        
        self.model_id = resolve_model_path(model_id)
        self.steps = steps
        self.device = device
        self.strength = strength
        self.scheduler_name = scheduler
        self.iterations = iterations
        self.pipe = None

        self._load_pipeline()

    def _load_pipeline(self):
        if self.pipe is not None:
            return

        print(f"   [DiffusionAttack] Loading {self.model_id} ({self.scheduler_name}) x{self.iterations}...")

        try:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16,
                safety_checker=None,           
                requires_safety_checker=False  
            ).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            self._set_scheduler(self.scheduler_name)
        except Exception as e:
            print(f"   [Error] Failed to load model {self.model_id}: {e}")
            self.pipe = None

    def _set_scheduler(self, scheduler_name):
        if self.pipe is None: return
        if scheduler_name not in SCHEDULER_MAP:
            print(f"   [Warning] Scheduler {scheduler_name} not found, using default.")
            return

        target_cls = SCHEDULER_MAP[scheduler_name]
        try:
            config = dict(self.pipe.scheduler.config)
            if scheduler_name == "DPM++ 2M SDE":
                config["algorithm_type"] = "sde-dpmsolver++"
                config["use_karras_sigmas"] = True
            elif scheduler_name == "DPM++ 2M":
                config["algorithm_type"] = "dpmsolver++"
            
            self.pipe.scheduler = target_cls.from_config(config)
        except Exception as e:
            print(f"   [Error] Failed to switch scheduler: {e}")
    
    def _cleanup(self):
        if hasattr(self, 'pipe') and self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def apply(self, image: Image.Image) -> Image.Image:
        self._load_pipeline()
        
        if self.pipe is None:
            print("   [Error] Pipeline not loaded, returning original image.")
            return image

        if image.mode != 'RGB': image = image.convert('RGB')
        
        current_img = image
        prompt = ""
        
        try:
            for i in range(self.iterations):
                with torch.no_grad():
                    out_img = self.pipe(
                        prompt=prompt, 
                        image=current_img, 
                        strength=self.strength, 
                        num_inference_steps=self.steps,
                        guidance_scale=1,
                        safety_checker=None,
                        requires_safety_checker=False,
                    ).images[0]
                current_img = out_img
        except Exception as e:
            print(f"   [Error] Diffusion Attack Failed: {e}")
            import traceback
            traceback.print_exc()
            return current_img 
        finally:
            self._cleanup()
            
        return current_img

    def get_param_str(self):
        prefix = f"Rinse{self.iterations}X" if self.iterations > 1 else "Diff"
        return f"{prefix}_{self.scheduler_name}_s{self.strength}"
    
@ATTACKS.register("Rinse-2xDiff")
class Rinse2XDiff(DiffusionRegeneration):
    def __init__(self, **kwargs):
        kwargs['iterations'] = 2
        super().__init__(**kwargs)

@ATTACKS.register("Rinse-4xDiff")
class Rinse4XDiff(DiffusionRegeneration):
    def __init__(self, **kwargs):
        kwargs['iterations'] = 4
        super().__init__(**kwargs)
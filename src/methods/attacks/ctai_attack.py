import torch
import torch.nn.functional as F
import numpy as np
import gc
from typing import Callable, List, Optional, Union
from PIL import Image

from diffusers import (
    StableDiffusionImg2ImgPipeline, 
    DDIMScheduler, 
    PNDMScheduler, 
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler, 
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler, 
    UniPCMultistepScheduler, 
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler, 
    KDPM2AncestralDiscreteScheduler, 
    LCMScheduler
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from src.core.interfaces import BaseAttack
from src.core.registry import ATTACKS
from src.core.paths import resolve_model_path

SCHEDULER_MAP = {
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
    "Euler": EulerDiscreteScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "LMS": LMSDiscreteScheduler,
    "UniPC": UniPCMultistepScheduler,
    "Heun": HeunDiscreteScheduler,
    "DPM2": KDPM2DiscreteScheduler,
    "DPM2 a": KDPM2AncestralDiscreteScheduler,
    "DPM++ 2M SDE": DPMSolverMultistepScheduler,
    "LCM": LCMScheduler,
}

class CTAIAttentionProcessor:
    def __init__(self):
        self.mode = "normal" 
        self.q_cache = []
        self.k_cache = []
        self.step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        is_self_attn = (encoder_hidden_states is None)
        query = attn.to_q(hidden_states)
        
        if is_self_attn:
            if self.mode == "save":
                key = attn.to_k(hidden_states)
                self.q_cache.append(query.clone().detach())
                self.k_cache.append(key.clone().detach())
                self.step += 1
            elif self.mode == "inject":
                
                idx = min(self.step, len(self.q_cache) - 1)
                query = self.q_cache[idx]
                key = self.k_cache[idx]
                self.step += 1
            else:
                key = attn.to_k(hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        
        value = attn.to_v(encoder_hidden_states if not is_self_attn and encoder_hidden_states is not None else hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(key.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(value.shape[0], -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], -1, attn.heads * head_dim)
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class CTAIPipeline(StableDiffusionImg2ImgPipeline):
    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, torch.FloatTensor],
        prompt: str = "",
        num_inference_steps: int = 50,
        strength: float = 0.7,   
        guidance_scale: float = 1.0, 
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        device = self._execution_device
        
        text_inputs = self.tokenizer([prompt], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(device))[0]

        if isinstance(image, Image.Image):
            w_orig, h_orig = image.size
            img_tensor = self.image_processor.preprocess(image).to(device, dtype=self.unet.dtype)
        else:
            img_tensor = image
            w_orig, h_orig = img_tensor.shape[-1], img_tensor.shape[-2]
            
        latents_0 = self.vae.encode(img_tensor).latent_dist.mean * self.vae.config.scaling_factor

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        original_attn_processors = self.unet.attn_processors
        ctai_processors = {name: CTAIAttentionProcessor() for name in self.unet.attn_processors.keys()}
        self.unet.set_attn_processor(ctai_processors)

        noise = torch.randn(latents_0.size(), generator=generator, device=device, dtype=latents_0.dtype)
        
        for p in ctai_processors.values():
            p.mode = "save"
            p.step = 0
            p.q_cache.clear()
            p.k_cache.clear()

        for t in timesteps:
            
            t_vec = t.unsqueeze(0) if torch.is_tensor(t) else torch.tensor([t], device=device)
            
            noisy_latents = self.scheduler.add_noise(latents_0, noise, t_vec)
            latent_model_input = self.scheduler.scale_model_input(noisy_latents, t)
            self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

        for p in ctai_processors.values():
            p.mode = "inject"
            p.step = 0

        t_start_vec = timesteps[0:1]
        latents = self.scheduler.add_noise(latents_0, noise, t_start_vec)

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for t in timesteps:
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                progress_bar.update()

        self.unet.set_attn_processor(original_attn_processors)
        image_attacked = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        image_attacked = (image_attacked / 2 + 0.5).clamp(0, 1)

        if output_type == "pil":
            image_attacked = image_attacked.cpu().permute(0, 2, 3, 1).numpy()
            image_attacked = self.numpy_to_pil(image_attacked)[0].resize((w_orig, h_orig))

        if not return_dict: return (image_attacked, False)
        return StableDiffusionPipelineOutput(images=[image_attacked], nsfw_content_detected=False)

@ATTACKS.register("CTAICloning")
class CTAICloningAttack(BaseAttack):
    def __init__(self, model_id="../../model/stable-diffusion-v1-5", steps=50, strength=0.7, scheduler_name="DDIM", device='cuda'):
        self.model_id = resolve_model_path(model_id)
        self.steps = steps
        self.strength = strength
        self.scheduler_name = scheduler_name
        self.device = device
        self.pipe = None
        self._load_pipeline()

    def _load_pipeline(self):
        if self.pipe: return
        print(f"\n   [CTAI] Loading Pipeline...")
        print(f"      - Model: {self.model_id}")
        print(f"      - Strength: {self.strength}")
        print(f"      - Scheduler: {self.scheduler_name}")
        
        self.pipe = CTAIPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16, safety_checker=None).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        target_cls = SCHEDULER_MAP.get(self.scheduler_name, DDIMScheduler)
        try:
            config = dict(self.pipe.scheduler.config)
            
            if self.scheduler_name == "DPM++ 2M SDE":
                config["algorithm_type"] = "sde-dpmsolver++"
                config["use_karras_sigmas"] = True
            elif self.scheduler_name == "DPM++ 2M":
                config["algorithm_type"] = "dpmsolver++"
                
            self.pipe.scheduler = target_cls.from_config(config)
        except Exception as e:
            print(f"   [CTAI Warning] Scheduler swap to {self.scheduler_name} failed: {e}. Fallback to DDIM.")
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def apply(self, image: Image.Image) -> Image.Image:
        self._load_pipeline()
        try:
            with torch.no_grad():
                return self.pipe(
                    image=image, 
                    prompt="", 
                    num_inference_steps=self.steps, 
                    strength=self.strength, 
                    guidance_scale=1.0
                ).images[0]
        except Exception as e:
            print(f"   [Error] CTAI Failed: {e}")
            import traceback
            traceback.print_exc()
            return image
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def get_param_str(self):
        model_short = self.model_id.split('/')[-1]
        return f"CTAI_{model_short}_{self.scheduler_name}_s{self.strength}_st{self.steps}"
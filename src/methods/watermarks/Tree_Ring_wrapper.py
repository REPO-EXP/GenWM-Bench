import torch
import numpy as np
import sys
import os
import math
import scipy.stats
from PIL import Image
from typing import Any, Dict, List, Union

from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

from src.core import BaseWatermark
from src.core.registry import WATERMARKS
from src.core.paths import resolve_model_path

@WATERMARKS.register("TreeRing")
class TreeRingWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        
        self.device = global_config.get('device', 'cuda')
        
        self.radius = self.config.get('radius', 10)
        self.channel = self.config.get('channel', 3)
        self.fixed_secret = self.config.get('fixed_secret', None)
        
        self.model_path = resolve_model_path(global_config.get('model_id') or self.config.get('model_path') or 'stabilityai/stable-diffusion-2-1-base')
        
        self.utils = None
        self.pipe = None 
        
        self.dummy_scheduler = None

    def _load_deps(self, pipeline=None):
        """ Pipeline"""
        if self.utils is not None and self.pipe is not None: 
            return
        
        try:
            
            sys.path.append(os.path.abspath('./src/watermark_method/Tree_ring'))
            from src.watermark_method.Tree_ring import circle_mask, get_pattern, get_noise
            self.utils = (circle_mask, get_pattern, get_noise)
            
            if pipeline is not None:
                self.model_path = getattr(pipeline.config, '_name_or_path', 'stabilityai/stable-diffusion-2-1-base')
            
            if self.model_path is None:
                self.model_path = 'stabilityai/stable-diffusion-2-1-base'

            try:
                self.dummy_scheduler = DDIMScheduler.from_pretrained(self.model_path, subfolder="scheduler")
                extract_scheduler = DDIMScheduler.from_pretrained(self.model_path, subfolder="scheduler")
            except:
                self.dummy_scheduler = DDIMScheduler()
                extract_scheduler = DDIMScheduler()

            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_path, 
                scheduler=extract_scheduler,
                dtype=torch.float16, 
                safety_checker=None
            ).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            
        except Exception as e:
            print(f"[TreeRing ERROR] Init failed: {e}")
            raise e

    def _transform_img(self, image):
        """Resize -> Normalize -> Tensor"""
        if not image.mode == "RGB": image = image.convert("RGB")
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.BICUBIC)
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(image).permute(2, 0, 1)

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        self._load_deps(pipeline)
        
        curr_pipe = pipeline if pipeline is not None else self.pipe
        circle_mask, get_pattern, get_noise = self.utils
        
        raw_seed = kwargs.get('seed', 42)
        if isinstance(raw_seed, list): seeds = [int(s) for s in raw_seed]
        else: seeds = [int(raw_seed)]
            
        if isinstance(prompt, list): prompts = prompt
        else: prompts = [str(prompt)] * len(seeds)
            
        target_secret = self.fixed_secret if self.fixed_secret is not None else secret
        if target_secret is not None:
            if isinstance(target_secret, list): w_seeds = [int(s) for s in target_secret]
            elif isinstance(target_secret, (int, float)): w_seeds = [int(target_secret)]
            elif torch.is_tensor(target_secret): w_seeds = [int(target_secret.sum().item())]
            else: w_seeds = [123456]
        else:
            w_seeds = [123456]

        min_len = min(len(prompts), len(seeds))
        if len(w_seeds) < min_len:
             w_seeds = w_seeds * (min_len // len(w_seeds)) + w_seeds[:min_len % len(w_seeds)]

        prompts = prompts[:min_len]
        seeds = seeds[:min_len]
        w_seeds = w_seeds[:min_len]
        batch_size = min_len

        shape_single = (1, 4, 64, 64)
        np_mask = circle_mask(shape_single[-1], r=self.radius)
        w_mask = torch.zeros(shape_single, dtype=torch.bool).to(self.device)
        w_mask[:, self.channel] = torch.tensor(np_mask).to(self.device)

        latents_list = []
        
        user_scheduler = curr_pipe.scheduler
        
        curr_pipe.scheduler = self.dummy_scheduler

        try:
            for i in range(batch_size):
                
                w_key = get_pattern(shape_single, curr_pipe, 1, w_seed=w_seeds[i]).to(self.device)
                
                gen = torch.Generator(device=self.device).manual_seed(seeds[i])
                
                init_latents = get_noise(curr_pipe, w_mask, w_key, 1, generator=gen)
                latents_list.append(init_latents)
        finally:
            
            curr_pipe.scheduler = user_scheduler
            
        batch_latents = torch.cat(latents_list, dim=0).to(self.device)
        batch_latents = batch_latents.to(dtype=curr_pipe.unet.dtype)

        gen_kwargs = {k: v for k, v in kwargs.items() if k not in ['seed', 'latents', 'generator', 'original_image']}
        
        with torch.no_grad():
            output = curr_pipe(
                prompts, 
                latents=batch_latents, 
                **gen_kwargs
            )
            
        return output.images

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        self._load_deps()
        circle_mask, get_pattern, _ = self.utils
        
        images = image if isinstance(image, list) else [image]
        batch_size = len(images)
        
        target_secret = self.fixed_secret if self.fixed_secret is not None else secret
        if target_secret is not None:
            if isinstance(target_secret, list): w_seeds = [int(s) for s in target_secret]
            elif isinstance(target_secret, (int, float)): w_seeds = [int(target_secret)]
            elif torch.is_tensor(target_secret): w_seeds = [int(target_secret.sum().item())]
            else: w_seeds = [123456]
        else: w_seeds = [123456]
        
        if len(w_seeds) < batch_size:
            w_seeds = w_seeds * (batch_size // len(w_seeds)) + w_seeds[:batch_size % len(w_seeds)]

        imgs_t_list = []
        dtype = self.pipe.unet.dtype 
        for img in images:
            t = self._transform_img(img).unsqueeze(0).to(self.device, dtype=dtype)
            imgs_t_list.append(t)
        batch_img_t = torch.cat(imgs_t_list, dim=0)

        original_scheduler = self.pipe.scheduler
        try:
            self.pipe.scheduler = DDIMInverseScheduler.from_config(original_scheduler.config)
        except:
            self.pipe.scheduler = DDIMInverseScheduler()
        
        try:
            with torch.no_grad():
                
                batch_img_t = batch_img_t.to(self.pipe.vae.dtype)
                scaling_factor = self.pipe.vae.config.scaling_factor
                image_latents = self.pipe.vae.encode(batch_img_t).latent_dist.mode() * scaling_factor
                
                inverted_latents = self.pipe(
                    prompt=[""] * batch_size, 
                    latents=image_latents, 
                    guidance_scale=1, 
                    num_inference_steps=50, 
                    output_type="latent"
                ).images 
        finally:
            
            self.pipe.scheduler = original_scheduler

        results = []
        shape_single = (1, 4, 64, 64)
        np_mask = circle_mask(shape_single[-1], r=self.radius)
        w_mask = torch.zeros(shape_single, dtype=torch.bool).to(self.device)
        w_mask[:, self.channel] = torch.tensor(np_mask).to(self.device)
        mask_bool = w_mask.squeeze(0) 

        for i in range(batch_size):
            try:
                curr_latent = inverted_latents[i]
                
                w_key = get_pattern(shape_single, self.pipe, 1, w_seed=w_seeds[i]).to(self.device)
                target_key = w_key.squeeze(0)

                curr_latent_float = curr_latent.float()
                fft = torch.fft.fftshift(torch.fft.fft2(curr_latent_float), dim=(-1, -2))
                
                feat = fft[mask_bool].flatten()
                target = target_key[mask_bool].flatten()
                
                feat = torch.cat([feat.real, feat.imag])
                target = torch.cat([target.real, target.imag])

                sigma = feat.std()
                lamda = (target ** 2 / sigma ** 2).sum().item()
                x = (((feat - target) / sigma) ** 2).sum().item()
                p_value = scipy.stats.ncx2.cdf(x=x, df=len(target), nc=lamda)
                
                safe_p = max(p_value, 1e-30)
                
                log_score = -math.log10(safe_p)
                
                threshold = 0.01
                if safe_p <= threshold:
                    
                    denominator = math.log(5.0 / safe_p, 10)
                    if denominator > 0:
                        conf_score = max(0.0, 1.0 - 1.0 / denominator)
                    else:
                        conf_score = 0.0
                else:
                    conf_score = 0.0

                results.append({
                    'p_value': float(p_value),
                    'score': float(conf_score), 
                    'log_score': float(log_score) 
                })

            except Exception as e:
                print(f"[TreeRing WARNING] Metric calc failed for img {i}: {e}")
                results.append({'p_value': 1.0, 'score': 0.0, 'log_score': 0.0})
            
        return results

    def compute_aggregate_metrics(self, all_sample_results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not all_sample_results: return {}
        
        p_values = np.array([r.get('p_value', 1.0) for r in all_sample_results])
        scores = np.array([r.get('score', 0.0) for r in all_sample_results])
        
        metrics = {
            'avg_p_value': float(np.mean(p_values)),
            'avg_conf_score': float(np.mean(scores)), 
        }
        
        threshold = 0.00555
        
        metrics['TPR@1e-2'] = float(np.mean(p_values < threshold))
        metrics['TPR@TreeRing'] = float(np.mean(p_values < threshold))
        
        metrics['sim_bit_acc'] = metrics['avg_conf_score']
        
        return metrics
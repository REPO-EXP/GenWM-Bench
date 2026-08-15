import argparse
import yaml
import os
import logging
import shutil
import numpy as np
from PIL import Image 
import gc
import time
import sys
from typing import Any, Dict, List, Union

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from diffusers import DDIMScheduler
from datasets import load_dataset
from diffusers.utils.torch_utils import randn_tensor

from src.core import BaseWatermark
from src.core.registry import WATERMARKS
from src.core.paths import resolve_model_path

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr_ssim(images, wm_images):
    """
     PSNR  SSIM
    """
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    for orig_img, wm_img in zip(images, wm_images):
        if not isinstance(orig_img, np.ndarray): orig_img = np.array(orig_img)
        if not isinstance(wm_img, np.ndarray): wm_img = np.array(wm_img)

        if orig_img.dtype == np.uint8 or orig_img.max() > 1.1:
            d_range = 255
        else:
            d_range = 1.0

        try:
            psnr_val = psnr(orig_img, wm_img, data_range=d_range)
        except Exception:
            psnr_val = 0.0
        total_psnr += psnr_val
        
        is_multichannel = (orig_img.ndim == 3)
        try:
            if is_multichannel:
                ssim_val = ssim(orig_img, wm_img, data_range=d_range, channel_axis=2)
            else:
                ssim_val = ssim(orig_img, wm_img, data_range=d_range)
        except TypeError:
            if is_multichannel:
                ssim_val = ssim(orig_img, wm_img, data_range=d_range, multichannel=True)
            else:
                ssim_val = ssim(orig_img, wm_img, data_range=d_range)
        except Exception:
            ssim_val = 0.0
            
        total_ssim += ssim_val
        count += 1
    
    if count == 0:
        return 0.0, 0.0

    return total_psnr / count, total_ssim / count

def binary_search_theta(gt_img_tensor, wm_img_tensor, threshold, lower=0., upper=1., precision=1e-6, max_iter=1000):
    
    gt_img_np = gt_img_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    wm_img_np = wm_img_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    
    optimal_theta = lower
    for i in range(max_iter):
        mid_theta = (lower + upper) / 2
        mixed_img_np = (gt_img_np - wm_img_np) * mid_theta + wm_img_np
        
        try:
            current_ssim = ssim(gt_img_np, mixed_img_np, channel_axis=-1, data_range=1.0)
        except TypeError:
            current_ssim = ssim(gt_img_np, mixed_img_np, multichannel=True, data_range=1.0)
        
        if current_ssim < threshold:
            lower = mid_theta
        else:
            upper = mid_theta
            optimal_theta = mid_theta
            
        if upper - lower < precision:
            break
    return optimal_theta

@WATERMARKS.register("ZoDiac")
class ZoDiacWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        
        self.device = torch.device(global_config.get('device', 'cuda'))
        
        if 'model_path' in self.config:
            self.model_path = resolve_model_path(self.config['model_path'])
        else:
            model_cfg = global_config.get('model_config', {})
            self.model_path = resolve_model_path(model_cfg.get('model_id', 'stabilityai/stable-diffusion-2-1-base'))
        
        self.num_inference_steps = self.config.get('num_inference_steps', 50)
        print(f"[ZoDiac Config] Model: {self.model_path} | Steps: {self.num_inference_steps}")

        self.w_type = self.config.get('w_type', 'single')
        self.w_channel = self.config.get('w_channel', 3)
        self.w_radius = self.config.get('w_radius', 10)
        self.w_seed = self.config.get('w_seed', 10)
        self.w_settings = self.config.get('w_settings', {}) 
        
        self.iters = self.config.get('iters', 100)
        self.loss_weights = self.config.get('loss_weights', [10.0, 0.1, 1.0, 0.0])
        self.lr = self.config.get('lr', 0.01)
        self.empty_prompt = self.config.get('empty_prompt', True)
        self.ssim_threshold = self.config.get('ssim_threshold', 0.92)
        self.detect_threshold = self.config.get('detect_threshold', 0.9)
        
        self.pipe = None
        self.wm_pipe = None
        self.loss_provider = None
        self.utils = None 

    def _load_deps(self):
        if self.pipe is not None: return
        
        print(f"[ZoDiac Wrapper] Loading Pipeline from: {self.model_path}")
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) 
            
            wrapper_path = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(wrapper_path)))) 
            zodiac_path = os.path.join(project_root, 'src', 'watermark_method', 'ZoDiac')
            
            if os.path.exists(zodiac_path) and zodiac_path not in sys.path:
                sys.path.insert(0, zodiac_path)
            
            from src.watermark_method.ZoDiac.main.wmdiffusion import WMDetectStableDiffusionPipeline
            from src.watermark_method.ZoDiac.main.wmpatch import GTWatermark, GTWatermarkMulti
            from src.watermark_method.ZoDiac.main.utils import get_img_tensor, save_img, compute_psnr, watermark_prob
            from src.watermark_method.ZoDiac.loss.loss import LossProvider
            
            self.utils = {
                'get_img_tensor': get_img_tensor,
                'save_img': save_img,
                'compute_psnr': compute_psnr,
                'watermark_prob': watermark_prob
            }
            
            try:
                scheduler = DDIMScheduler.from_pretrained(self.model_path, subfolder="scheduler")
            except Exception as e:
                print(f"[ZoDiac Warning] Failed to load scheduler: {e}")
                scheduler = DDIMScheduler()

            self.pipe = WMDetectStableDiffusionPipeline.from_pretrained(
                self.model_path, 
                scheduler=scheduler,
                torch_dtype=torch.float16, 
                safety_checker=None,
                image_encoder=None,
                num_inference_steps=self.num_inference_steps, 
            )
            self.pipe.set_progress_bar_config(disable=True)
            
            self.pipe.vae.requires_grad_(False)
            self.pipe.text_encoder.requires_grad_(False)
            self.pipe.unet.requires_grad_(False)
            
            self.pipe = self.pipe.to(self.device)

            self.pipe.unet.enable_gradient_checkpointing()
            if hasattr(self.pipe, 'text_encoder'):
                try: self.pipe.text_encoder.gradient_checkpointing_enable()
                except: pass
            
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            
            try: self.pipe.enable_xformers_memory_efficient_attention()
            except: pass
            
            if self.w_type == 'single':
                self.wm_pipe = GTWatermark(
                    self.device, w_channel=self.w_channel, w_radius=self.w_radius, 
                    generator=torch.Generator(self.device).manual_seed(self.w_seed)
                )
            elif self.w_type == 'multi':
                self.wm_pipe = GTWatermarkMulti(
                    self.device, w_settings=self.w_settings, 
                    generator=torch.Generator(self.device).manual_seed(self.w_seed)
                )
            
            self.loss_provider = LossProvider(self.loss_weights, self.device)
            print(f"[ZoDiac Wrapper] Loaded successfully (GPU Mode).")
            
        except Exception as e:
            print(f"[ZoDiac ERROR] Init failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _transform_img(self, image):
        if image is None: return None
        if image.mode != 'RGB': image = image.convert("RGB")
        image = image.resize((512, 512))
        return torch.from_numpy(np.array(image)).float().div(255).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float16)

    def _get_init_latent(self, img_tensor, text_embeddings):
        with torch.no_grad():
            with torch.autocast("cuda"): 
                img_latents = self.pipe.get_image_latents(img_tensor, sample=False)
                reversed_latents = self.pipe.forward_diffusion(
                    latents=img_latents,
                    text_embeddings=text_embeddings,
                    guidance_scale=1.0,
                    num_inference_steps=self.num_inference_steps,
                )
        return reversed_latents

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        self._load_deps()
        from tqdm.auto import tqdm
        
        images = kwargs.get('original_image', [])
        if not isinstance(images, list): images = [images] if images is not None else []
        
        prompts = prompt if isinstance(prompt, list) else [str(prompt)]
        raw_seed = kwargs.get('seed', 42)
        seeds = [int(s) for s in raw_seed] if isinstance(raw_seed, list) else [int(raw_seed)]
        
        target_len = max(len(prompts), len(images), len(seeds))
        
        def broadcast(lst, length, default=None):
            if not lst: return [default] * length
            if len(lst) >= length: return lst[:length]
            return (lst * (length // len(lst) + 1))[:length]

        images = broadcast(images, target_len, None)
        prompts = broadcast(prompts, target_len, "")
        seeds = broadcast(seeds, target_len, 42)
        
        output_images = []
        compute_psnr = self.utils['compute_psnr']
        
        with torch.no_grad():
            with torch.autocast("cuda"):
                empty_text_embeddings = self.pipe.get_text_embedding('')
        
        total_start_time = time.time()
        
        for i in range(target_len):
            img_pil = images[i]
            cur_prompt = prompts[i]
            cur_seed = seeds[i]
            
            torch.cuda.empty_cache()
            gc.collect()

            if img_pil is None:
                print(f"[ZoDiac Info] Generating GT for Image {i} (Seed: {cur_seed})...")
                generator = torch.Generator(self.device).manual_seed(cur_seed)
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        clean_out = self.pipe(
                            cur_prompt, 
                            num_inference_steps=self.num_inference_steps, 
                            output_type='pil',
                            generator=generator
                        )
                    img_pil = clean_out.images[0]

            try:
                print(f"\n[ZoDiac] Optimizing Image {i+1}/{target_len}")
                
                gt_img_tensor = self._transform_img(img_pil)
                init_latents_approx = self._get_init_latent(gt_img_tensor, empty_text_embeddings)
                
                init_latents = init_latents_approx.detach().clone().float()
                init_latents.requires_grad = True
                
                optimizer = optim.Adam([init_latents], lr=self.lr)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.3)
                
                progress_bar = tqdm(range(self.iters), desc="Optimization", leave=False)
                
                for step in progress_bar:
                    with torch.autocast("cuda"):
                        init_latents_wm = self.wm_pipe.inject_watermark(init_latents)
                        
                        pred_img_tensor = self.pipe(
                            '', guidance_scale=1.0, 
                            num_inference_steps=self.num_inference_steps, 
                            output_type='tensor', use_trainable_latents=True, 
                            init_latents=init_latents_wm
                        ).images
                        
                        loss = self.loss_provider(pred_img_tensor, gt_img_tensor, init_latents_wm, self.wm_pipe)
                    
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    if step % 10 == 0:
                        with torch.no_grad():
                            curr_psnr = compute_psnr(pred_img_tensor.float(), gt_img_tensor.float())
                            progress_bar.set_postfix({'L': f"{loss.item():.2f}", 'PSNR': f"{curr_psnr:.1f}"})
                
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        init_latents_wm = self.wm_pipe.inject_watermark(init_latents)
                        wm_img_tensor = self.pipe(
                            '', guidance_scale=1.0, 
                            num_inference_steps=self.num_inference_steps, 
                            output_type='tensor', use_trainable_latents=True, 
                            init_latents=init_latents_wm
                        ).images
                
                optimal_theta = binary_search_theta(
                    gt_img_tensor.float(), 
                    wm_img_tensor.float(), 
                    self.ssim_threshold, 
                    precision=0.01
                )
                
                final_tensor = (gt_img_tensor.float() - wm_img_tensor.float()) * optimal_theta + wm_img_tensor.float()
                final_tensor = torch.clamp(final_tensor, 0, 1)
                
                final_np = (final_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt_np = (gt_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                final_psnr, final_ssim = calculate_psnr_ssim([Image.fromarray(gt_np)], [Image.fromarray(final_np)])
                print(f"    [Result] Image {i} | Theta: {optimal_theta:.4f} | PSNR: {final_psnr:.2f} dB | SSIM: {final_ssim:.4f}")
                
                output_images.append(Image.fromarray(final_np))
                
            except Exception as e:
                print(f"[ZoDiac ERROR] Embed failed for image {i}: {e}")
                import traceback
                traceback.print_exc()
                if img_pil is not None:
                    output_images.append(img_pil.resize((512,512)))
                else:
                    output_images.append(Image.new('RGB', (512, 512)))
        
        total_end_time = time.time()
        count = len(output_images) if output_images else 1
        avg_time = (total_end_time - total_start_time) / count
        print(f"\n[ZoDiac Info] Batch Finished. Count: {count}")
        print(f" > Total Time: {total_end_time - total_start_time:.2f} s")
        print(f" > Average Embedding Time: {avg_time:.4f} s/image")
        
        return output_images

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        self._load_deps()
        images = image if isinstance(image, list) else [image]
        results = []
        
        with torch.no_grad():
            with torch.autocast("cuda"):
                tester_prompt = '' 
                text_embeddings = self.pipe.get_text_embedding(tester_prompt)
            
        watermark_prob = self.utils['watermark_prob']

        for img in images:
            if img is None:
                results.append({'score': 0.0})
                continue
            
            try:
                img_tensor = self._transform_img(img)
                with torch.autocast("cuda"): 
                    det_prob = 1 - watermark_prob(img_tensor, self.pipe, self.wm_pipe, text_embeddings)
                score = float(det_prob)
            except Exception as e:
                print(f"[ZoDiac ERROR] Extract failed: {e}")
                score = 0.0
            
            results.append({'score': score})
            
        return results

    def compute_aggregate_metrics(self, all_sample_results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not all_sample_results: return {}
        scores = np.array([r['score'] for r in all_sample_results])
        metrics = {'avg_prob': float(np.mean(scores))}
        metrics['TPR@0.9p*threshod'] = float(np.mean(scores > self.detect_threshold))
        return metrics
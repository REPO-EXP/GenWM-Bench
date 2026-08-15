import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Union
from src.core import BaseWatermark
import sys
import os
import traceback
from src.core.registry import WATERMARKS
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

@WATERMARKS.register("ROBIN")
class ROBINWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        
        self.device = global_config.get('device', 'cuda')
        self.pipe = None
        self.utils = None
        
        self.wm_path_v2 = self.config.get('wm_path')      
        self.wm_path_v15 = self.config.get('wm_pathV1.5') 
        
        self.threshold_v15 = self.config.get('threshold_v15', 68.8) 
        self.threshold_v2 = self.config.get('threshold_v2', 65.0)
        self.current_threshold = self.threshold_v15 
        
        self.current_loaded_path = None
        self.opt_wm = None
        self.opt_acond = None
        
        self.w_seed = self.config.get('w_seed', 999999) 
        self.w_pattern = self.config.get('w_pattern', 'rand')
        self.w_mask_shape = self.config.get('w_mask_shape', 'circle') 
        self.w_radius = self.config.get('w_radius', 10)
        self.w_up_radius = self.config.get('w_up_radius', 10)
        self.w_low_radius = self.config.get('w_low_radius', 5)
        self.w_channel = self.config.get('w_channel', 3)
        self.w_measurement = self.config.get('w_measurement', 'l1_complex')
        self.w_injection = self.config.get('w_injection', 'complex')
        self.w_pattern_const = self.config.get('w_pattern_const', 0)
        self.fixed_secret = self.config.get('fixed_secret', None)
        
        self.image_length = self.config.get('image_length', 512)
        self.injection_steps = self.config.get('injection_steps', 35)
        self.num_inference_steps = self.config.get('num_inference_steps', 50)
        self.test_num_inference_steps = self.config.get('test_num_inference_steps', 50)
        self.guidance_scale = self.config.get('guidance_scale', 7.5)
        self.lguidance = self.config.get('lguidance', 7.5)
        
        self._ensure_paths()

    def _ensure_paths(self):
        try:
            if not os.path.exists('logs'): os.makedirs('logs', exist_ok=True)
            robin_path = os.path.abspath('./src/watermark_method/ROBIN')
            if robin_path not in sys.path:
                sys.path.append(robin_path)
            
            from src.watermark_method.ROBIN.optim_utils import eval_watermark, get_watermarking_mask
            self.utils = {'get_mask': get_watermarking_mask, 'eval_wm': eval_watermark}
            
        except Exception as e:
            print(f"[ROBIN ERROR] Dependency setup failed: {e}")

    def _load_wm_weights(self, target_path):
        """"""
        if self.current_loaded_path == target_path and self.opt_wm is not None:
            return 
            
        if not target_path or not os.path.exists(target_path):
            raise FileNotFoundError(f"[ROBIN] Watermark file not found: {target_path}")
            
        print(f"[ROBIN] Loading watermark weights from: {target_path}")
        wm_data = torch.load(target_path, map_location=self.device)
        self.opt_wm = wm_data['opt_wm'].to(self.device)
        self.opt_acond = wm_data['opt_acond'].to(self.device)
        self.current_loaded_path = target_path

        if target_path == self.wm_path_v15:
            self.current_threshold = self.threshold_v15
            print(f"[ROBIN] Detected V1.5 Weights. Switching Threshold -> {self.current_threshold}")
        elif target_path == self.wm_path_v2:
            self.current_threshold = self.threshold_v2
            print(f"[ROBIN] Detected V2 Weights. Switching Threshold -> {self.current_threshold}")

    def _wrap_pipeline(self, original_pipe):
        """ SD Pipeline  Inversable Pipeline"""
        from src.watermark_method.ROBIN.inverse_stable_diffusion import InversableStableDiffusionPipeline
        
        if isinstance(original_pipe, InversableStableDiffusionPipeline):
            return original_pipe

        if not isinstance(original_pipe.scheduler, DPMSolverMultistepScheduler):
            scheduler = DPMSolverMultistepScheduler.from_config(original_pipe.scheduler.config)
        else:
            scheduler = original_pipe.scheduler

        new_pipe = InversableStableDiffusionPipeline(
            vae=original_pipe.vae,
            text_encoder=original_pipe.text_encoder,
            tokenizer=original_pipe.tokenizer,
            unet=original_pipe.unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        new_pipe = new_pipe.to(self.device)
        if hasattr(original_pipe, 'dtype'):
            new_pipe.to(dtype=original_pipe.dtype)
        new_pipe.set_progress_bar_config(disable=True)
        return new_pipe

    def _transform_img(self, img):
        if img.mode != 'RGB': img = img.convert("RGB")
        if img.size != (self.image_length, self.image_length):
            img = img.resize((self.image_length, self.image_length), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        
        self.pipe = self._wrap_pipeline(pipeline)
        
        text_enc_dim = self.pipe.text_encoder.config.hidden_size
        target_wm_path = self.wm_path_v15 if text_enc_dim == 768 else self.wm_path_v2
        if text_enc_dim == 1024: target_wm_path = self.wm_path_v2
            
        if not target_wm_path:
            
            target_wm_path = self.config.get('wm_path')
            
        if not target_wm_path:
            print(f"❌ [ROBIN] Unsupported model dimension: {text_enc_dim}.")
            return [None] * (len(prompt) if isinstance(prompt, list) else 1)
            
        try:
            self._load_wm_weights(target_wm_path)
        except Exception as e:
            print(f"❌ [ROBIN] Failed to load weights: {e}")
            return [None] * (len(prompt) if isinstance(prompt, list) else 1)

        self.opt_acond = self.opt_acond.to(dtype=self.pipe.text_encoder.dtype)

        raw_seed = kwargs.get('seed', 42)
        if isinstance(raw_seed, list): seeds = [int(s) for s in raw_seed]
        else: seeds = [int(raw_seed)]
        prompts = prompt if isinstance(prompt, list) else [str(prompt)]
        
        target_len = max(len(prompts), len(seeds))
        
        def broadcast_list(lst, length):
            if len(lst) == 0: return []
            if len(lst) >= length: return lst[:length]
            repeat = (length // len(lst)) + 1
            return (lst * repeat)[:length]

        prompts = broadcast_list(prompts, target_len)
        seeds = broadcast_list(seeds, target_len)
        generators = [torch.Generator(self.device).manual_seed(s) for s in seeds]
        batch_size = target_len

        latents_list = []
        for i in range(batch_size):
            l = self.pipe.get_random_latents(
                height=self.image_length, width=self.image_length, generator=generators[i]
            )
            latents_list.append(l)
        init_latents = torch.cat(latents_list, dim=0).to(self.device, dtype=self.pipe.unet.dtype)

        watermarking_mask = self.utils['get_mask'](init_latents, self, self.device)
        
        try:
            gt_patch_expanded = self.opt_wm.repeat(batch_size, 1, 1, 1)
            opt_acond_expanded = self.opt_acond.repeat(batch_size, 1, 1)

            with torch.no_grad():
                outputs = self.pipe(
                    prompt=prompts,
                    latents=init_latents,
                    generator=generators, 
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    height=self.image_length,
                    width=self.image_length,
                    watermarking_mask=watermarking_mask,
                    watermarking_steps=self.injection_steps,
                    gt_patch=gt_patch_expanded,   
                    opt_acond=opt_acond_expanded,
                    lguidance=self.lguidance,
                    args=self 
                )
            return outputs.images
        except Exception as e:
            print(f"[ROBIN ERROR] Embed failed: {e}")
            traceback.print_exc()
            return [None] * batch_size

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        self._ensure_paths()
        
        pipe_candidate = kwargs.get('pipeline', self.pipe)
        
        if pipe_candidate is None:
            
            print("[ROBIN] Error: No pipeline available for extraction.")
            return [{'score': 100.0}] * (len(image) if isinstance(image, list) else 1)

        self.pipe = self._wrap_pipeline(pipe_candidate)
        
        if not isinstance(self.pipe.scheduler, DPMSolverMultistepScheduler):
             self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        ori_scheduler = self.pipe.scheduler
        
        hidden_size = self.pipe.text_encoder.config.hidden_size
        target_path = self.wm_path_v2 if hidden_size == 1024 else self.wm_path_v15
        
        try:
            self._load_wm_weights(target_path)
        except Exception as e:
            print(f"[ROBIN Extract] Failed to load weights: {e}")
            return [{'score': 100.0}] * (len(image) if isinstance(image, list) else 1)

        self.opt_acond = self.opt_acond.to(dtype=self.pipe.unet.dtype)

        images = image if isinstance(image, list) else [image]
        results = []
        
        text_embeddings = self.pipe.get_text_embedding("")

        for img in images:
            if img is None:
                results.append({'score': 100.0})
                continue
            try:
                img_tensor = self._transform_img(img).unsqueeze(0).to(self.device).to(self.pipe.unet.dtype)
                
                with torch.no_grad():
                    
                    image_latents = self.pipe.get_image_latents(img_tensor, sample=False)
                
                latents_b = [image_latents]
                with torch.no_grad():
                    
                    _, latents_b, _ = self.pipe.forward_diffusion(
                        latents=image_latents,
                        text_embeddings=text_embeddings,
                        guidance_scale=1.0, 
                        num_inference_steps=self.test_num_inference_steps,
                        latents_b=latents_b
                    )
                
                if len(latents_b) > self.injection_steps:
                    latent_at_step = latents_b[self.injection_steps]
                    current_mask = self.utils['get_mask'](latent_at_step, self, self.device)
                    _, w_metric = self.utils['eval_wm'](latent_at_step, latent_at_step, current_mask, self.opt_wm, self)
                    score = float(w_metric)
                else:
                    score = 100.0 
            except Exception as e:
                print(f"[ROBIN Extract Error] {e}")
                
                traceback.print_exc()
                score = 100.0
                
            results.append({'score': score})

        self.pipe.scheduler = ori_scheduler
        return results

    def compute_aggregate_metrics(self, all_sample_results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not all_sample_results: return {}
        scores = np.array([r['score'] for r in all_sample_results])
        
        tpr = float(np.mean(scores < self.current_threshold))
        
        return {
            'avg_loss': float(np.mean(scores)),
            f'TPR (thresh={self.current_threshold})': tpr
        }
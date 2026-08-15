import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Union
from src.core import BaseWatermark
import sys
from src.core.registry import WATERMARKS
from src.core.paths import resolve_model_path

@WATERMARKS.register("GaussianShading")
class GaussianShadingWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        
        d = global_config.get('device', 'cuda')
        self.device = torch.device(d) if isinstance(d, str) else d
        
        self.pipe = None
        self.gs_module = None
        self._load_pipeline()

    def _load_pipeline(self):
        sys.path.append('./src/watermark_method/Gaussian_Shading')
        from src.watermark_method.Gaussian_Shading import Gaussian_Shading_chacha, InversableStableDiffusionPipeline
        from diffusers import DPMSolverMultistepScheduler
        
        model_path = resolve_model_path(self.config['model_path'])
        try:
            scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')
        except:
            
            scheduler = DPMSolverMultistepScheduler()

        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            safety_checker=None, 
            requires_safety_checker=False 
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        self.gs_module = Gaussian_Shading_chacha(
            self.config['channel_copy'], self.config['hw_copy'], 
            float(self.config.get('fpr', 1e-6)), self.config['user_number']
        )

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        from src.watermark_method.Gaussian_Shading import set_random_seed

        raw_seed = kwargs.get('seed', 42)
        if isinstance(raw_seed, list): seeds = [int(s) for s in raw_seed]
        else: seeds = [int(raw_seed)]
            
        if isinstance(prompt, list): prompts = prompt
        else: prompts = [str(prompt)] * len(seeds)
            
        min_len = min(len(prompts), len(seeds))
        prompts = prompts[:min_len]
        seeds = seeds[:min_len]

        s_base = (secret if torch.is_tensor(secret) else torch.tensor(secret)).to(self.device)
        
        c_dim = int(4 // self.gs_module.ch)
        h_dim = int(64 // self.gs_module.hw)
        s_reshaped = s_base.reshape(1, c_dim, h_dim, h_dim)

        latents_list = []
        for i in range(len(seeds)):
            set_random_seed(int(seeds[i])) 
            wm_latent = self.gs_module.create_watermark_and_return_w(s_reshaped)
            latents_list.append(wm_latent.to(self.device, dtype=self.pipe.unet.dtype))
            
        batch_latents = torch.cat(latents_list, dim=0)
        
        with torch.no_grad():
            output = self.pipe(
                prompts, 
                latents=batch_latents, 
                num_inference_steps=kwargs.get('num_inference_steps', 25), 
                guidance_scale=kwargs.get('guidance_scale', 7.5)
            )
        
        return output.images

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        from src.watermark_method.Gaussian_Shading import transform_img
            
        images = image if isinstance(image, list) else [image]
        batch_size = len(images)

        if secret is not None and self.gs_module.key is None:
            try:
                
                s_base = (secret if torch.is_tensor(secret) else torch.tensor(secret)).to(self.device)
                c_dim = int(4 // self.gs_module.ch)
                h_dim = int(64 // self.gs_module.hw)
                s_reshaped = s_base.reshape(1, c_dim, h_dim, h_dim)
                
                _ = self.gs_module.create_watermark_and_return_w(s_reshaped)
                
            except Exception as e:
                print(f"[GaussianShading] Warning: Key init failed: {e}")
        
        imgs_t_list = []
        for img in images:
            t = transform_img(img).unsqueeze(0).to(self.device, dtype=self.pipe.unet.dtype)
            imgs_t_list.append(t)
        batch_img_t = torch.cat(imgs_t_list, dim=0)

        latents = self.pipe.get_image_latents(batch_img_t, sample=False)
        
        try:
            emb = self.pipe.get_text_embedding([''] * batch_size)
        except:
             
             emb = self._get_text_embedding_fallback([''] * batch_size)

        inv_latents = self.pipe.forward_diffusion(
            latents=latents, text_embeddings=emb, guidance_scale=1, num_inference_steps=20
        )
        
        rev_m_batch = (inv_latents > 0).int()
        
        target = None
        if secret is not None:
            target = (secret.cpu() if torch.is_tensor(secret) else torch.tensor(secret)).numpy().flatten()

        results = []
        
        for i in range(batch_size):
            try:
                single_rev_m = rev_m_batch[i].flatten().cpu().numpy()
                
                rev_sd = self.gs_module.stream_key_decrypt(single_rev_m)
                rev_wm = self.gs_module.diffusion_inverse(rev_sd)
                pred_bits = rev_wm.flatten().tolist()
                
                metrics = {'raw_bits': pred_bits}
                
                if target is not None:
                    min_l = min(len(pred_bits), len(target))
                    acc = (np.array(pred_bits[:min_l]) == target[:min_l]).mean()
                    metrics['bit_acc'] = float(acc)
                
                results.append(metrics)
            except Exception as e:
                print(f"[GaussianShading] Extract failed for item {i}: {e}")
                results.append({'bit_acc': 0.0})
            
        return results

    def _get_text_embedding_fallback(self, prompts):
        
        text_inputs = self.pipe.tokenizer(
            prompts, padding="max_length", max_length=self.pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        return self.pipe.text_encoder(text_input_ids)[0]
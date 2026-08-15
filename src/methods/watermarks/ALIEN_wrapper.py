import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Union, Optional
import sys
import os
import io
import torch.nn.functional as F
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

from src.core import BaseWatermark
from src.core.registry import WATERMARKS
from src.core.paths import resolve_model_path

@WATERMARKS.register("ALIEN")
class ALIENWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        
        self.device = torch.device(global_config.get('device', 'cuda'))
        
        self.encoder_path = self.config.get('encoder_path', 'data/models/ALIEN/encoder.pth')
        self.decoder_path = self.config.get('decoder_path', 'data/models/ALIEN/decoder.pth')

        self.secret_len = self.config.get('secret_config', {}).get('length', 48)
        self.latent_channels = self.config.get('latent_channels', 4)
        
        self.wm_weight = float(self.config.get('wm_weight', 1.0)) 
        
        self.injection_start = self.config.get('wm_injection_start_step', 20)
        self.injection_end = self.config.get('wm_injection_end_step', 45)
        self.default_steps = self.config.get('num_inference_steps', 50)
        
        self.wm_encoder = None
        self.wm_decoder = None
        self.vae = None
        self._model_dtype = torch.float32

        self.LatentMarkEncoder = None
        self.LatentMarkDecoder = None
        self.WatermarkInjectionPipeline = None

    def _load_deps(self):
        if self.WatermarkInjectionPipeline is not None: return

        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

        try:
            from src.watermark_method.ALIEN.model import LatentMarkEncoder, LatentMarkDecoder
            from src.watermark_method.ALIEN.pipeline import WatermarkInjectionPipeline
            
            self.LatentMarkEncoder = LatentMarkEncoder
            self.LatentMarkDecoder = LatentMarkDecoder
            self.WatermarkInjectionPipeline = WatermarkInjectionPipeline
        except ImportError as e:
            print(f"[ALIEN ERROR] Failed to import ALIEN modules: {e}")
            raise e

    def _load_models(self):
        if self.wm_encoder is not None: return
        self._load_deps()

        self.wm_encoder = self.LatentMarkEncoder(secret_size=self.secret_len, latent_channels=self.latent_channels)
        self.wm_decoder = self.LatentMarkDecoder(latent_channels=self.latent_channels, secret_size=self.secret_len)
        
        if not os.path.exists(self.encoder_path) or not os.path.exists(self.decoder_path):
            raise FileNotFoundError(f"[ALIEN] Model weights not found at {self.encoder_path} or {self.decoder_path}")

        state_dict_enc = torch.load(self.encoder_path, map_location='cpu')
        state_dict_dec = torch.load(self.decoder_path, map_location='cpu')
        self.wm_encoder.load_state_dict(state_dict_enc)
        self.wm_decoder.load_state_dict(state_dict_dec)

        self.wm_encoder.to(self.device).to(dtype=self._model_dtype).eval()
        self.wm_decoder.to(self.device).to(dtype=self._model_dtype).eval()

    def _wrap_pipeline(self, original_pipe):
        """
         SD Pipeline  Pipeline
        """
        scheduler = original_pipe.scheduler
            
        new_pipe = self.WatermarkInjectionPipeline(
            vae=original_pipe.vae,
            text_encoder=original_pipe.text_encoder,
            tokenizer=original_pipe.tokenizer,
            unet=original_pipe.unet,
            scheduler=scheduler, 
            safety_checker=None,
            feature_extractor=getattr(original_pipe, 'feature_extractor', None),
            image_encoder=getattr(original_pipe, 'image_encoder', None),
            requires_safety_checker=False,
            wm_encoder=self.wm_encoder,
            wm_decoder=self.wm_decoder
        )
        new_pipe = new_pipe.to(self.device)
        new_pipe.set_progress_bar_config(disable=True)
        return new_pipe

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        self._load_models()

        target_dtype = pipeline.unet.dtype
        if self._model_dtype != target_dtype:
            
            self._model_dtype = target_dtype
            self.wm_encoder.to(dtype=target_dtype)
            self.wm_decoder.to(dtype=target_dtype)
        
        prompts = prompt if isinstance(prompt, list) else [prompt]
        batch_size = len(prompts)
        
        if secret is None:
            
            secret_tensor = torch.randint(0, 2, (1, self.secret_len), dtype=torch.float32, device=self.device)
            secret_tensor = secret_tensor.repeat(batch_size, 1)
        else:
            
            if isinstance(secret, list): 
                secret_tensor = torch.tensor(secret, dtype=torch.float32, device=self.device)
            elif isinstance(secret, np.ndarray): 
                secret_tensor = torch.from_numpy(secret).float().to(self.device)
            elif torch.is_tensor(secret): 
                secret_tensor = secret.float().to(self.device)
            else:
                
                secret_tensor = torch.randint(0, 2, (1, self.secret_len), dtype=torch.float32, device=self.device)
        
        if secret_tensor.dim() == 1: 
            secret_tensor = secret_tensor.unsqueeze(0)
        
        if secret_tensor.shape[0] == 1 and batch_size > 1:
            secret_tensor = secret_tensor.repeat(batch_size, 1)
        
        if secret_tensor.shape[0] != batch_size:
            
            repeat_factor = batch_size // secret_tensor.shape[0] + 1
            secret_tensor = secret_tensor.repeat(repeat_factor, 1)[:batch_size]
            
        secret_tensor = secret_tensor[:, :self.secret_len]
        
        wm_pipe = self._wrap_pipeline(pipeline)
        
        self.vae = pipeline.vae

        steps = kwargs.get('num_inference_steps', self.default_steps)
        input_seed = kwargs.get('seed', 42)
        
        seeds = []
        if isinstance(input_seed, int):
            seeds = [input_seed + i for i in range(batch_size)]
        elif isinstance(input_seed, list):
            
            seeds = (input_seed * (batch_size // len(input_seed) + 1))[:batch_size]
        else:
            seeds = [42] * batch_size
             
        try:
            
            generators = [torch.Generator(self.device).manual_seed(s) for s in seeds]
            
            output_alien = wm_pipe(
                prompt=prompts, 
                secret_input=secret_tensor,
                wm_injection_start_step=self.injection_start,
                wm_injection_end_step=self.injection_end,
                wm_weight=self.wm_weight,
                num_inference_steps=steps,
                guidance_scale=kwargs.get('guidance_scale', 7.5),
                enable_watermark=True,
                output_type="pil",
                
                generator=generators, 
                height=kwargs.get('height', 512),
                width=kwargs.get('width', 512)
            )
            
            if isinstance(output_alien, dict):
                images = output_alien["images"]
            else:
                images = output_alien.images
            
            self._cached_secret = secret_tensor.cpu().detach()
            return images
            
        except Exception as e:
            print(f"[ALIEN ERROR] Embed failed: {e}")
            import traceback
            traceback.print_exc()
            return [None] * batch_size

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        self._load_models()
        images = image if isinstance(image, list) else [image]
        results = []
        
        vae = self.vae
        if vae is None:
            
            vae = AutoencoderKL.from_pretrained(resolve_model_path("../../model/stable-diffusion-v1-5"), subfolder="vae").to(self.device)
        
        vae = vae.to(dtype=self._model_dtype).eval()
        
        target_secret = None
        if secret is not None:
            if isinstance(secret, list):
                target_secret = torch.tensor(secret, dtype=self._model_dtype, device=self.device)
            elif isinstance(secret, np.ndarray):
                target_secret = torch.from_numpy(secret).to(self.device, dtype=self._model_dtype)
            elif torch.is_tensor(secret):
                target_secret = secret.to(self.device, dtype=self._model_dtype)
        elif hasattr(self, '_cached_secret'):
            target_secret = self._cached_secret.to(self.device).to(dtype=self._model_dtype)
            
        if target_secret is not None:
            if target_secret.dim() == 1: target_secret = target_secret.unsqueeze(0)
            
            if target_secret.shape[0] == 1 and len(images) > 1:
                target_secret = target_secret.repeat(len(images), 1)
            
            elif target_secret.shape[0] != len(images):
                 repeat = len(images) // target_secret.shape[0] + 1
                 target_secret = target_secret.repeat(repeat, 1)[:len(images)]

        to_tensor = transforms.ToTensor()
        
        for idx, img in enumerate(images):
            if img is None:
                results.append({'bit_acc': 0.0, 'error': 'None image'})
                continue

            try:
                img_tensor = to_tensor(img).unsqueeze(0).to(self.device, dtype=self._model_dtype)
                img_tensor = img_tensor * 2.0 - 1.0 
                
                with torch.no_grad():
                    
                    dist = vae.encode(img_tensor).latent_dist
                    latents = dist.sample() * vae.config.scaling_factor
                    
                    pred_vals = self.wm_decoder(latents)
                
                curr_pred = pred_vals.view(-1)
                pred_bits = (curr_pred > 0.5).float()
                
                metrics = {
                    'raw_bits': pred_bits.cpu().tolist()
                }

                if target_secret is not None:
                    curr_target = target_secret[idx].view(-1)
                    min_len = min(len(curr_pred), len(curr_target))
                    
                    acc = (pred_bits[:min_len] == curr_target[:min_len]).float().mean().item()
                    metrics['bit_acc'] = float(acc)
                else:
                    metrics['bit_acc'] = 0.0 
                
                results.append(metrics)
                
            except Exception as e:
                print(f"[ALIEN ERROR] Extract failed for img {idx}: {e}")
                results.append({'bit_acc': 0.5}) 
        
        return results
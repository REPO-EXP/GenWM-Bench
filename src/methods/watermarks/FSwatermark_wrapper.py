
import torch
import numpy as np
from PIL import Image
from src.core import BaseWatermark
import sys
import os

try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        if hasattr(huggingface_hub, 'hf_hub_download'):
            huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except ImportError:
    pass

from src.core.registry import WATERMARKS
from src.core.paths import resolve_model_path

@WATERMARKS.register("FSWatermark")
class FSWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        self.device = global_config.get('device', 'cuda')
        self.vae = None
        self.encoder = None
        self.decoder = None

    def _load(self):
        if self.vae is not None: return

        try:
            
            from src.watermark_method.FSwatermark import SEncoder, SDecoder
            from src.watermark_method.FSwatermark.diffusers import AutoencoderKL
            
            self.encoder = SEncoder(secret_size=self.config.get('secret_length', 100), image_size=512).to(self.device)
            self.decoder = SDecoder(secret_size=self.config.get('secret_length', 100), image_size=512).to(self.device)
            self.encoder.load_state_dict(torch.load(self.config['encoder_path'], map_location=self.device))
            self.decoder.load_state_dict(torch.load(self.config['decoder_path'], map_location=self.device))
            
            self.vae = AutoencoderKL.from_pretrained(resolve_model_path(self.config['model_path']), subfolder="vae").to(self.device)
            self.vae.load_state_dict(torch.load(self.config['watermarked_vae_path'], map_location=self.device))
            self.vae.eval()
            
        except ImportError as e:
            raise ImportError(f" 'src/watermark_method/FSwatermark' \n: {e}")

    def embed(self, pipeline, prompt, secret, **kwargs):
        self._load()
        
        raw_seed = kwargs.get('seed', 42)
        seeds = [int(s) for s in raw_seed] if isinstance(raw_seed, list) else [int(raw_seed)]
        
        prompts = prompt if isinstance(prompt, list) else [str(prompt)] * len(seeds)
        
        s_base = (secret if torch.is_tensor(secret) else torch.tensor(secret)).float().to(self.device)
        s_batch = s_base.unsqueeze(0).repeat(len(seeds), 1) if s_base.dim() == 1 else s_base[:len(seeds)]
        
        generators = [torch.Generator(self.device).manual_seed(s) for s in seeds]

        with torch.no_grad():
            _, secret_v = self.encoder(s_batch, None)
            
            if kwargs.get('original_image') is None:
                latents = pipeline(prompts, generator=generators, num_inference_steps=50, output_type="latent").images
                
                latents = latents / self.vae.config.scaling_factor
                
            else:
                imgs = kwargs.get('original_image')
                if not isinstance(imgs, list): imgs = [imgs]
                imgs_t = [torch.from_numpy(np.array(im.convert("RGB")).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(self.device)*2-1 for im in imgs]
                latents = self.vae.encode(torch.cat(imgs_t)).latent_dist.sample()

            recon = self.vae.decode(latents, secret_v, able=1).sample
            
            recon = (recon / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
            
            return [Image.fromarray((recon[i] * 255).astype(np.uint8)) for i in range(len(recon))]

    def extract(self, image, secret=None, **kwargs):
        self._load()
        images = image if isinstance(image, list) else [image]
        
        imgs_t = [torch.from_numpy(np.array(im.convert("RGB")).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(self.device) * 2.0 - 1.0 for im in images]
        
        with torch.no_grad():
            preds = self.decoder(torch.cat(imgs_t))
            pred_bits = (preds > 0.5).float().cpu().numpy()

        if secret is not None:
            target = (secret.cpu() if torch.is_tensor(secret) else torch.tensor(secret)).numpy().flatten()
            return {'bit_acc': float((pred_bits == target[None, :]).mean())}
        else:
            return [{'raw_bits': pred_bits[0].tolist()}]
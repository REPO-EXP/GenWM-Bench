import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from typing import Any, Dict, List, Union
from src.core import BaseWatermark
import sys
import os
from src.core.registry import WATERMARKS
import traceback

@WATERMARKS.register("StegaStamp")
class StegaStampWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        
        self.encoder_net = None
        self.decoder_net = None
        self.device = global_config.get('device', 'cuda')
        self.image_size = (400, 400) 

    def _load_models(self):
        if self.encoder_net is not None: return
        
        print("[StegaStamp] Loading Models...")
        try:
            sys.path.append(os.path.abspath('./src/watermark_method/Stegastamp'))
            from src.watermark_method.StegaStamp import StegaStampEncoder, StegaStampDecoder

            self.encoder_net = StegaStampEncoder().to(self.device)
            self.decoder_net = StegaStampDecoder().to(self.device)
            
            enc_path = self.config['encoder_path']
            dec_path = self.config['decoder_path']
            
            self.encoder_net.load_state_dict(torch.load(enc_path, map_location=self.device))
            self.decoder_net.load_state_dict(torch.load(dec_path, map_location=self.device))
            
            self.encoder_net.eval()
            self.decoder_net.eval()
            print("[StegaStamp] Models loaded successfully.")
            
        except Exception as e:
            print(f"[StegaStamp ERROR] Model loading failed: {str(e)}")
            traceback.print_exc()
            raise e

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        try:
            self._load_models()
            
            raw_seed = kwargs.get('seed', 42)
            if isinstance(raw_seed, list):
                seeds = [int(s) for s in raw_seed]
            else:
                seeds = [int(raw_seed)]

            prompts = prompt if isinstance(prompt, list) else [str(prompt)]
            
            if len(prompts) == 1 and len(seeds) > 1:
                prompts = prompts * len(seeds)
            
            min_len = min(len(seeds), len(prompts))
            seeds = seeds[:min_len]
            prompts = prompts[:min_len]
            
            generators = [torch.Generator(self.device).manual_seed(s) for s in seeds]

            clean_images = []
            original_images = kwargs.get('original_image')

            if original_images is not None:
                
                if not isinstance(original_images, list):
                    clean_images = [original_images]
                else:
                    clean_images = original_images
                for idx in range(len(clean_images)):
                    if clean_images[idx] is not None and clean_images[idx].size != self.image_size:
                        clean_images[idx] = clean_images[idx].resize(self.image_size, Image.BICUBIC)
            else:
                with torch.no_grad():
                    out = pipeline(prompts, generator=generators, **kwargs)
                    clean_images = [img.resize(self.image_size, Image.BICUBIC) for img in out.images]
            
            if len(clean_images) == 0: return []

            out_dir = kwargs.get('output_dir', '')
            if out_dir:
                cd = os.path.join(out_dir, 'clean_images'); os.makedirs(cd, exist_ok=True)
                for ci, cimg in enumerate(clean_images):
                    if hasattr(cimg, 'save'): cimg.save(os.path.join(cd, f"sample_{ci}.png"))

            batch_size = len(clean_images)
            
            if torch.is_tensor(secret):
                s_tensor = secret.float().to(self.device)
            elif isinstance(secret, (list, np.ndarray)):
                s_tensor = torch.tensor(secret).float().to(self.device)
            else:

                s_tensor = torch.randint(0, 2, (1, 100)).float().to(self.device)

            if s_tensor.dim() == 1:
                s_tensor = s_tensor.unsqueeze(0)
            if s_tensor.size(0) == 1 and batch_size > 1:
                s_tensor = s_tensor.repeat(batch_size, 1)
            
            output_images = []
            to_tensor = transforms.ToTensor()
            
            for i, img in enumerate(clean_images):
                img_tensor = to_tensor(img).unsqueeze(0).to(self.device)
                
                current_secret = s_tensor[i].unsqueeze(0)
                
                with torch.no_grad():
                    residual = self.encoder_net((current_secret, img_tensor))
                    encoded = torch.clamp(img_tensor + residual, 0, 1)
                
                encoded_pil = transforms.ToPILImage()(encoded.squeeze(0).cpu())
                output_images.append(encoded_pil)

            return output_images

        except Exception as e:
            print(f"[StegaStamp ERROR] Embed failed: {str(e)}")
            traceback.print_exc()
            return []

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        try:
            self._load_models()
            
            images = image if isinstance(image, list) else [image]
            
            targets = []
            if secret is not None:
                if torch.is_tensor(secret):
                    secret_np = secret.cpu().numpy()
                else:
                    secret_np = np.array(secret)
                
                if secret_np.ndim == 2 and secret_np.shape[0] == len(images):
                    targets = list(secret_np)
                else:
                    targets = [secret_np.flatten()] * len(images)
            else:
                targets = [None] * len(images)

            results = []
            to_tensor = transforms.ToTensor()
            
            for i, img in enumerate(images):
                img_resized = ImageOps.fit(img.convert("RGB"), self.image_size)
                img_tensor = to_tensor(img_resized).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    
                    logits = self.decoder_net(img_tensor)
                    preds = (logits > 0.5).float()
                
                pred_bits = preds.cpu().numpy().flatten().astype(int).tolist()
                
                metrics = {'raw_bits': pred_bits}
                
                if targets[i] is not None:
                    tgt = targets[i].flatten()
                    min_len = min(len(pred_bits), len(tgt))
                    acc = (np.array(pred_bits[:min_len]) == tgt[:min_len]).mean()
                    metrics['bit_acc'] = float(acc)
                
                results.append(metrics)
                
            return results

        except Exception as e:
            print(f"[StegaStamp ERROR] Extract failed: {str(e)}")
            traceback.print_exc()
            return [{'bit_acc': 0.0, 'error': str(e)}] * (len(image) if isinstance(image, list) else 1)
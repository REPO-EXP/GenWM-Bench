
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms 
from omegaconf import OmegaConf
from src.core import BaseWatermark
import sys
import os
from typing import Dict, List, Any, Optional, Union
from src.core.registry import WATERMARKS
import importlib
import traceback

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__': return None
        elif config == "__is_unconditional__": return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

@WATERMARKS.register("RoSteALS")
class RoSteALS(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        self.model = None
        self.device = global_config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'

    def _load_model(self):
        if self.model is not None: return
        sys.path.insert(0, './src/watermark_method/RoSteALS')
        
        config_path = self.config.get('rosteals_config_path')
        ckpt_path = self.config.get('rosteals_ckpt_path')
        
        print(f"[RoSteALS] Loading config: {config_path}")
        
        conf = OmegaConf.load(config_path).model
        length = self.config.get('secret_config', {}).get('length', 100)
        conf.params.control_config.params.secret_len = length
        conf.params.decoder_config.params.secret_len = length
        
        model = instantiate_from_config(conf)
        try:
            sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            sd = torch.load(ckpt_path, map_location='cpu')

        if 'state_dict' in sd:
            sd = sd['state_dict']
            
        model.load_state_dict(sd, strict=False)
        self.model = model.to(self.device).eval()
        print("[RoSteALS] Model loaded successfully.")

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        try:
            self._load_model()
            
            raw_seed = kwargs.get('seed', 42)
            seeds = [int(s) for s in raw_seed] if isinstance(raw_seed, list) else [int(raw_seed)]
            prompts = prompt if isinstance(prompt, list) else [str(prompt)] * len(seeds)
            min_len = min(len(seeds), len(prompts))
            seeds, prompts = seeds[:min_len], prompts[:min_len]
            generators = [torch.Generator(self.device).manual_seed(s) for s in seeds]

            image_input = kwargs.get('original_image')
            if image_input is None:
                
                clean_images = pipeline(prompts, generator=generators).images
            else:
                clean_images = image_input if isinstance(image_input, list) else [image_input]
            
            if len(clean_images) == 0: return []

            if isinstance(secret, list):
                s_tensor = torch.tensor(secret, dtype=torch.float).to(self.device)
            elif isinstance(secret, np.ndarray):
                 s_tensor = torch.from_numpy(secret).float().to(self.device)
            elif torch.is_tensor(secret):
                s_tensor = secret.float().to(self.device)
            else:
                s_tensor = torch.randint(0, 2, (len(clean_images), 100)).float().to(self.device)

            if s_tensor.dim() == 1:
                s_tensor = s_tensor.unsqueeze(0).repeat(len(clean_images), 1)
            elif s_tensor.dim() == 2 and s_tensor.shape[0] == 1:
                s_tensor = s_tensor.repeat(len(clean_images), 1)

            tform = transforms.Compose([
                transforms.Resize((256, 256)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            output_images = []
            
            for i, cover_org in enumerate(clean_images):
                w, h = cover_org.size
                cover = tform(cover_org).unsqueeze(0).to(self.device)
                current_s = s_tensor[i].unsqueeze(0)

                with torch.no_grad():
                    z = self.model.encode_first_stage(cover)
                    z_embed, _ = self.model(z, None, current_s)
                    stego = self.model.decode_first_stage(z_embed)
                    
                    res = stego.clamp(-1, 1) - cover
                    res = torch.nn.functional.interpolate(res, (h, w), mode='bilinear')
                    res_np = res.permute(0, 2, 3, 1).cpu().detach().numpy()[0]

                img_normalized = np.array(cover_org).astype(float) / 127.5 - 1.0
                stego_float = np.clip(img_normalized + res_np, -1, 1)
                stego_uint8 = (stego_float * 127.5 + 127.5).astype(np.uint8)
                
                output_images.append(Image.fromarray(stego_uint8))
                
            return output_images

        except Exception as e:
            print(f"[RoSteALS ERROR] Embed failed: {str(e)}")
            traceback.print_exc()
            return []

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        try:
            self._load_model()
            images = image if isinstance(image, list) else [image]
            
            targets = []
            if secret is not None:
                if torch.is_tensor(secret) and secret.dim() > 1 and secret.shape[0] == len(images):
                    targets = secret.cpu().numpy()
                elif isinstance(secret, (list, np.ndarray)) and len(secret) == len(images) and isinstance(secret[0], (list, np.ndarray)):
                    targets = np.array(secret)
                else:
                    tgt = np.array(secret).flatten() if not torch.is_tensor(secret) else secret.cpu().numpy().flatten()
                    targets = [tgt] * len(images)
            else:
                targets = [None] * len(images)

            tform = transforms.Compose([
                transforms.Resize((256, 256)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            results = []
            
            for i, img in enumerate(images):
                img_t = tform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    z = self.model.encode_first_stage(img_t)
                    stego = self.model.decode_first_stage(z)
                    bits = (self.model.decoder(stego) > 0).float().cpu().numpy().flatten().tolist()
                
                metrics = {'raw_bits': bits}
                
                if targets[i] is not None:
                    tgt = np.array(targets[i]).flatten()
                    pred = np.array(bits)
                    min_len = min(len(pred), len(tgt))
                    acc = (pred[:min_len] == tgt[:min_len]).mean()
                    metrics['bit_acc'] = float(acc)
                
                results.append(metrics)

            return results
            
        except Exception as e:
            print(f"[RoSteALS ERROR] Extract failed: {str(e)}")
            traceback.print_exc()
            return [{'bit_acc': 0.0, 'error': str(e)}] * (len(image) if isinstance(image, list) else 1)
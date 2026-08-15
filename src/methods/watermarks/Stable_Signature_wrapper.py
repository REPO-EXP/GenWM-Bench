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
from src.core.paths import resolve_model_path
import importlib
import traceback
from diffusers import AutoencoderKL

def _fix_paths():
    current_path = os.path.abspath(".")
    base_method_path = os.path.join(current_path, "src", "watermark_method", "Stable_Signature")

    local_open_clip_src = os.path.join(base_method_path, "open_clip", "src")
    if os.path.exists(local_open_clip_src):
        if local_open_clip_src not in sys.path:
            
            sys.path.append(local_open_clip_src)
    
    possible_ldm_roots = [base_method_path, os.path.join(base_method_path, "src")]
    ldm_found = False
    for p in possible_ldm_roots:
        if os.path.exists(os.path.join(p, "ldm")):
            
            sys.path.append(p)
            ldm_found = True
            break
            
    if not ldm_found:
        for root, dirs, files in os.walk(base_method_path):
            if "ldm" in dirs:
                
                if root not in sys.path:
                    sys.path.append(root)
                break
_fix_paths()

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

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    try:
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        pl_sd = torch.load(ckpt, map_location="cpu")
        
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def process(image):
    if not image.mode == "RGB":
        image = image.convert("RGB")
    
    image = image.resize((512, 512), resample=Image.Resampling.BICUBIC)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image

def torch_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images.detach().cpu().float() / 2 + 0.5).clamp(0, 1)
    images = images.permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype(np.uint8)
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

@WATERMARKS.register("Stable_Signature")
class StableSignature(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        
        self.device = global_config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
        
        self.ldm_aef = None      
        self.msg_extractor = None 
        
        self.key = '111010110101000001010111010011010100010000100111'
        self.fixed_key_bits = np.array(list(map(int, self.key)))

    def _load_model(self):
        if self.ldm_aef is not None: return

        print(f"[Stable_Signature] Loading Models...")
        try:
            ldm_config_path = resolve_model_path(self.config['ldm_config'])
            ldm_ckpt_path = resolve_model_path(self.config['ldm_ckpt'])
            
            print(f"    -> Building LDM from {ldm_config_path}")
            config = OmegaConf.load(ldm_config_path)
            
            if hasattr(config.model.params, "cond_stage_config"):
                config.model.params.cond_stage_config = "__is_unconditional__"
            
            ldm_ae = load_model_from_config(config, ldm_ckpt_path)
            self.ldm_aef = ldm_ae.first_stage_model
            self.ldm_aef.to(self.device).eval()
            
            wm_vae_path = self.config['watermarked_vae_path']
            print(f"    -> Loading Watermark: {wm_vae_path}")
            try:
                state_dict = torch.load(wm_vae_path, map_location=self.device, weights_only=False)
            except TypeError:
                state_dict = torch.load(wm_vae_path, map_location=self.device)
            
            self.ldm_aef.load_state_dict(state_dict, strict=False)
            print(f"    -> Weights injected.")

            ext_path = self.config['extractor_path']
            print(f"    -> Loading Extractor: {ext_path}")
            self.msg_extractor = torch.jit.load(ext_path).to(self.device).eval()
            
        except Exception as e:
            print(f"[Stable_Signature ERROR] Loading failed: {str(e)}")
            traceback.print_exc()
            raise e

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
                clean_images = pipeline(prompts, generator=generators, **kwargs).images
            else:
                
                clean_images = image_input if isinstance(image_input, list) else [image_input]
            
            if len(clean_images) == 0: return []

            output_images = []
            
            vae_encoder = pipeline.vae
            
            for img in clean_images:
                
                target_dtype = vae_encoder.dtype
                img_t = process(img).unsqueeze(0).to(self.device).to(target_dtype)
                
                with torch.no_grad():
                    
                    latent = vae_encoder.encode(img_t).latent_dist.sample()
                    
                    res = self.ldm_aef.decode(latent.to(self.ldm_aef.dtype))
                    
                wm_img = torch_to_pil(res)[0]
                output_images.append(wm_img)
            
            return output_images

        except Exception as e:
            print(f"[Stable_Signature ERROR] Embed failed: {str(e)}")
            traceback.print_exc()
            return []

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        try:
            self._load_model()
            images = image if isinstance(image, list) else [image]
            
            target = self.fixed_key_bits
            
            targets = [target] * len(images)

            norm_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            results = []
            
            for i, img in enumerate(images):
                if img.mode != 'RGB': img = img.convert('RGB')
                
                img_t = norm_transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits = self.msg_extractor(img_t)
                    pred_bits = (logits > 0).float().cpu().numpy().flatten().tolist()
                
                metrics = {'raw_bits': pred_bits}
                
                if targets[i] is not None:
                    tgt = targets[i]
                    pred = np.array(pred_bits)
                    min_len = min(len(pred), len(tgt))
                    acc = (pred[:min_len] == tgt[:min_len]).mean()
                    metrics['bit_acc'] = float(acc)
                
                results.append(metrics)
            
            return results
            
        except Exception as e:
            print(f"[Stable_Signature ERROR] Extract failed: {str(e)}")
            return [{'bit_acc': 0.0, 'error': str(e)}] * (len(image) if isinstance(image, list) else 1)
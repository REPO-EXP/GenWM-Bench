import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import traceback
import importlib
from PIL import Image
from typing import Any, Dict, List, Union
import safetensors.torch as _sft
from torchvision import transforms
from diffusers import UNet2DConditionModel, AutoencoderKL

from src.core import BaseWatermark
from src.core.registry import WATERMARKS
from src.core.paths import resolve_model_path

@WATERMARKS.register("SleeperMark")
class SleeperMark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        super().__init__(kwargs, global_config)
        
        self.device = global_config.get('device', 'cuda')
        self.wm_unet = None
        self.extractor = None
        self.vae_model = None 
        self.gt_secret = None

    def _ensure_deps(self):
        
        current_dir = os.getcwd()
        
        target_path = os.path.join(current_dir, "src", "watermark_method", "SleeperMark", "Stage2")
        
        for p in [os.path.dirname(target_path), target_path]:
            if os.path.exists(p) and not os.path.exists(os.path.join(p, "__init__.py")):
                with open(os.path.join(p, "__init__.py"), 'w') as f: pass

        try:
            module_path = "src.watermark_method.SleeperMark.Stage2.watermarkModel"
            return importlib.import_module(module_path)
        except ImportError as e:
            
            try:
                return importlib.import_module("src.watermark_method.SleeperMark.watermarkModel")
            except:
                raise ImportError(f"SleeperMark : {e}")

    def _load_models(self):
        
        if self.extractor is not None:
            return

        print(f"[SleeperMark] Initializing Components...")
        wm_model_mod = self._ensure_deps()

        try:
            
            unet_dir = self.config.get('unet_dir')
            print(f"    -> Loading UNet from {unet_dir}")
            self.wm_unet = UNet2DConditionModel.from_pretrained(unet_dir).to(self.device)
            
            pretrained_dir = self.config.get('pretrained_wm_dir')
            secret_path = os.path.join(pretrained_dir, "secret.pt")
            if os.path.exists(secret_path):
                self.gt_secret = torch.load(secret_path, map_location='cpu').to(self.device)

            self.extractor = wm_model_mod.Extractor_forLatent(secret_size=self.gt_secret.shape[0])
            decoder_pth = os.path.join(pretrained_dir, "decoder.pth")
            self.extractor.load_state_dict(torch.load(decoder_pth, map_location=self.device))
            self.extractor.to(self.device).eval()

            base_model = resolve_model_path(self.config.get('base_model_path', '../../model/stable-diffusion-v1-5'))
            print(f"    -> Loading standalone VAE from {base_model}")
            self.vae_model = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(self.device)

        except Exception as e:
            print(f"[SleeperMark ERROR] Model loading failed: {e}")
            traceback.print_exc()

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any = None, **kwargs) -> List[Image.Image]:
        active_adapters = []
        original_unet = None
        try:
            self._load_models()
            if self.wm_unet is None:
                raise RuntimeError("Model loading failed, check config.json and LFS weights")

            target_dtype = pipeline.dtype

            original_unet = pipeline.unet

            self.wm_unet = self.wm_unet.to(dtype=target_dtype)
            pipeline.unet = self.wm_unet

            pipeline.safety_checker = None
            pipeline.requires_safety_checker = False

            extra_lora_path = kwargs.get('extra_lora_path', self.config.get('extra_lora_path'))
            extra_lora_scale = kwargs.get('extra_lora_scale', self.config.get('extra_lora_scale', 0.8))

            if extra_lora_path and os.path.exists(extra_lora_path):
                print(f"[SleeperMark] Loading extra LoRA: {extra_lora_path} (scale={extra_lora_scale})")
                _wn = os.path.basename(extra_lora_path)
                try:
                    pipeline.load_lora_weights(extra_lora_path, weight_name=_wn, adapter_name="sleeper_style")
                    active_adapters.append("sleeper_style")
                    pipeline.set_adapters(active_adapters, adapter_weights=[extra_lora_scale])
                except Exception:
                    from src.core.lora_utils import load_lora_and_merge
                    self._lora_saved = load_lora_and_merge(pipeline.unet, extra_lora_path, scale=extra_lora_scale)

            trigger = self.config.get('trigger', '*[Z]& ')
            if isinstance(prompt, list):
                modified_prompts = [trigger + p for p in prompt]
            else:
                modified_prompts = trigger + str(prompt)

            raw_seed = kwargs.get('seed', 42)
            if isinstance(raw_seed, list):
                generators = [torch.Generator(self.device).manual_seed(s) for s in raw_seed]
            else:
                generators = torch.Generator(self.device).manual_seed(int(raw_seed))

            gen_kwargs = {k: v for k, v in kwargs.items() if k not in ['seed', 'original_image', 'extra_lora_path', 'extra_lora_scale']}

            print(f"[SleeperMark] Generating images with trigger...")
            outputs = pipeline(
                prompt=modified_prompts,
                generator=generators,
                **gen_kwargs
            )

            images = outputs.images
            if images and isinstance(images[0], list):
                images = [img for batch in images for img in batch]
            return images

        except Exception as e:
            print(f"[SleeperMark ERROR] Embed failed: {e}")
            traceback.print_exc()
            return []
        finally:
            
            if hasattr(self, '_lora_saved') and self._lora_saved is not None:
                try:
                    from src.core.lora_utils import unmerge_lora
                    unmerge_lora(pipeline.unet, self._lora_saved)
                except:
                    pass
                self._lora_saved = None
            if active_adapters:
                try:
                    pipeline.delete_adapters(active_adapters)
                except:
                    try:
                        pipeline.unload_lora_weights()
                    except:
                        pass
            
            if original_unet is not None:
                pipeline.unet = original_unet

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        try:
            self._load_models()
            images = image if isinstance(image, list) else [image]
            
            pipeline = kwargs.get('pipeline')
            vae = pipeline.vae if pipeline is not None else self.vae_model
            
            results = []
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            for img in images:
                if img.mode != 'RGB': img = img.convert('RGB')
                
                img_t = transform(img).unsqueeze(0).to(self.device).to(vae.dtype)
                
                with torch.no_grad():
                    
                    latents = vae.encode(img_t).latent_dist.sample() * vae.config.scaling_factor
                    logits = self.extractor(latents)
                    decoded_secret = torch.sigmoid(logits)
                    
                    pred_bits = (decoded_secret > 0.5).float().detach().cpu().numpy().flatten()
                
                metrics = {'raw_bits': pred_bits.tolist()}
                
                if self.gt_secret is not None:
                    target = self.gt_secret.detach().cpu().numpy().flatten()
                    acc = (pred_bits == target).mean()
                    metrics['bit_acc'] = float(acc)
                
                results.append(metrics)
            
            return results

        except Exception as e:
            print(f"[SleeperMark ERROR] Extract failed: {e}")
            traceback.print_exc()
            return [{'bit_acc': 0.0}] * (len(image) if isinstance(image, list) else 1)
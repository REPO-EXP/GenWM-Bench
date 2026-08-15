from typing import Dict, List, Any, Union, Optional
import numpy as np
try: import cv2
except ImportError: cv2 = None
import torch
from PIL import Image
from src.core.registry import WATERMARKS 
from src.core import BaseWatermark
import sys
import random
import os

@WATERMARKS.register("ImWatermarkWrapper")
class ImWatermarkWrapper(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        
        self.device = global_config.get('device', 'cuda')
        self.algorithm_type = self.config.get('algorithm_type', 'dwtDct')
        self.scale = self.config.get('scale', 1.0)

    def _ensure_deps(self):
        try:
            try:
                import imwatermark
                from imwatermark import WatermarkEncoder, WatermarkDecoder
                return WatermarkEncoder, WatermarkDecoder
            except ImportError:
                current_dir = os.getcwd()
                target_path = os.path.join(current_dir, "src", "watermark_method", "imwatermark")
                if target_path not in sys.path:
                    sys.path.append(target_path)
                from src.watermark_method.imwatermark import WatermarkEncoder, WatermarkDecoder
                return WatermarkEncoder, WatermarkDecoder
        except ImportError as e:
            raise ImportError(f" ImWatermark: {e}")

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        
        Encoder, _ = self._ensure_deps()
        
        gen_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['seed', 'original_image', 'global_config']
        }
        
        if 'num_inference_steps' not in gen_kwargs: gen_kwargs['num_inference_steps'] = 50
        if 'guidance_scale' not in gen_kwargs: gen_kwargs['guidance_scale'] = 7.5
        
        raw_seed = kwargs.get('seed', 42)
        if isinstance(raw_seed, list): seeds = [int(s) for s in raw_seed]
        else: seeds = [int(raw_seed)]
            
        if isinstance(prompt, str): prompts = [prompt] * len(seeds)
        else: prompts = prompt
        
        min_len = min(len(prompts), len(seeds))
        prompts = prompts[:min_len]
        
        generators = [torch.Generator(self.device).manual_seed(s) for s in seeds[:min_len]]

        clean_images = []

        original = kwargs.get('original_image')
        if original is not None:
            clean_images = [original] if not isinstance(original, list) else list(original)
        elif pipeline is not None:
            try:
                clean_images = pipeline(prompts, generator=generators, **gen_kwargs).images
            except Exception as e:
                print(f"[ImWatermark] Generation failed: {e}")
                return []

        else:
            raise ValueError("Embed  pipeline  original_image")

        watermarked_images = []
        
        if torch.is_tensor(secret): 
            secret_bits = secret.cpu().numpy().astype(int).tolist()
        elif isinstance(secret, (list, np.ndarray)): 
            secret_bits = list(secret)
        else: 
            secret_bits = [0] * 32

        encoder = Encoder()
        encoder.set_watermark('bits', secret_bits)

        for i, img in enumerate(clean_images):
            try:
                current_seed = seeds[i]
                random.seed(current_seed)
                np.random.seed(current_seed)
                
                img_np = np.array(img.convert("RGB"))
                bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                bgr_encoded = encoder.encode(bgr, self.algorithm_type)
                
                if bgr_encoded is None: 
                    print(f"[Warning] Sample {i} encoding failed.")
                    watermarked_images.append(img)
                else:
                    wm_pil = Image.fromarray(cv2.cvtColor(bgr_encoded, cv2.COLOR_BGR2RGB))
                    watermarked_images.append(wm_pil)

            except Exception as e:
                print(f"[ImWatermark] Embed Error on sample {i}: {e}")
                watermarked_images.append(img)

        return watermarked_images

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        _, Decoder = self._ensure_deps()
        
        if isinstance(image, list): images_to_process = image
        else: images_to_process = [image]

        target = None
        if secret is not None:
            target = secret.cpu().numpy().flatten() if torch.is_tensor(secret) else np.array(secret).flatten()
            length = len(target)
        else:
            length = self.config.get('secret_length', 32)
            
        results = []
        for img in images_to_process:
            try:
                bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
                decoder = Decoder('bits', length)
                watermark = decoder.decode(bgr, self.algorithm_type)
                
                if watermark is None: watermark = [0] * length
                pred_bits = [1 if x else 0 for x in watermark]
                
                metrics = {'raw_bits': pred_bits}
                if target is not None:
                    
                    min_len = min(len(target), len(pred_bits))
                    error_bits = np.sum(target[:min_len] != np.array(pred_bits[:min_len]))
                    metrics['bit_acc'] = float(1.0 - (error_bits / min_len))
                else:
                    metrics['bit_acc'] = -1.0
                
                results.append(metrics)
                
            except Exception as e:
                results.append({'bit_acc': 0.0, 'error': str(e)})

        return results
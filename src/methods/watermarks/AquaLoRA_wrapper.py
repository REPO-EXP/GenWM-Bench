import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Union
from src.core import BaseWatermark
import sys
import os
from src.core.registry import WATERMARKS

@WATERMARKS.register("AquaLoRA")
class AquaLoRAWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)
        self.device = global_config.get('device', 'cuda')
    
    def _ensure_deps(self):
        
        current_dir = os.getcwd()
        target_path = os.path.join(current_dir, "src", "watermark_method", "AquaLoRA")
        folder_name = "AquaLoRA"
        
        if not os.path.exists(target_path):
            target_path_lower = os.path.join(current_dir, "src", "watermark_method", "Aqualora")
            if os.path.exists(target_path_lower):
                target_path = target_path_lower
                folder_name = "Aqualora"

        paths_to_check = [
            os.path.join(current_dir, "src", "watermark_method"),
            target_path,
            os.path.join(target_path, "scripts"),
            os.path.join(target_path, "evaluation")
        ]
        for p in paths_to_check:
            if os.path.exists(p) and not os.path.exists(os.path.join(p, "__init__.py")):
                with open(os.path.join(p, "__init__.py"), 'w') as f: pass

        try:
            module_base = f"src.watermark_method.{folder_name}"
            import importlib
            create_mod = importlib.import_module(f"{module_base}.scripts.create_wm_lora")
            eval_mod = importlib.import_module(f"{module_base}.evaluation.utils_eval")
            return create_mod.create_watermark_lora, eval_mod.simple_decode
        except ImportError as e:
            raise ImportError(f" AquaLoRA : {e}")

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        import tempfile
        import safetensors.torch as _sft

        create_wm_lora, _ = self._ensure_deps()

        extra_lora_path = kwargs.get('extra_lora_path', self.config.get('extra_lora_path', None))
        extra_lora_scale = kwargs.get('extra_lora_scale', self.config.get('extra_lora_scale', 0.8))

        raw_seed = kwargs.get('seed', 42)
        seeds = [int(s) for s in raw_seed] if isinstance(raw_seed, list) else [int(raw_seed)]
        generators = [torch.Generator(self.device).manual_seed(s) for s in seeds]

        if isinstance(prompt, str):
            prompts = [prompt] * len(seeds)
        else:
            prompts = prompt
        prompts = prompts[:min(len(prompts), len(seeds))]
        generators = generators[:min(len(prompts), len(seeds))]

        active_adapters = []
        adapter_weights = []
        _tmp_lora = None

        try:
            
            if extra_lora_path and os.path.exists(extra_lora_path):
                print(f"[AquaLoRA] Loading extra LoRA: {extra_lora_path}")
                _wn = os.path.basename(extra_lora_path)
                try:
                    pipeline.load_lora_weights(extra_lora_path, weight_name=_wn, adapter_name="aqua_style")
                    active_adapters.append("aqua_style")
                    adapter_weights.append(extra_lora_scale)
                except Exception:
                    
                    from src.core.lora_utils import load_lora_and_merge
                    self._lora_saved = load_lora_and_merge(pipeline.unet, extra_lora_path, scale=extra_lora_scale)
                    active_adapters = []  

            if torch.is_tensor(secret):
                s_list = secret.cpu().numpy().astype(int).flatten()
            else:
                s_list = np.array(secret).flatten()
            binary_str = ''.join(str(x) for x in s_list)

            _, lora_weights = create_wm_lora(
                self.config['aqualora_model_path'],
                scale=1.0,
                msg_bits=len(s_list),
                hidinfo=binary_str,
                save=False
            )

            _tmp_lora = tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False)
            _sft.save_file(lora_weights, _tmp_lora.name)
            _wn = os.path.basename(_tmp_lora.name)
            pipeline.load_lora_weights(_tmp_lora.name, weight_name=_wn, adapter_name="aqua_wm")
            active_adapters.append("aqua_wm")
            adapter_weights.append(1.0)

            pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)

            pipeline.safety_checker = None
            pipeline.requires_safety_checker = False

            gen_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ['seed', 'extra_lora_path', 'extra_lora_scale', 'original_image']}
            if 'num_inference_steps' not in gen_kwargs:
                gen_kwargs['num_inference_steps'] = 25

            output = pipeline(prompt=prompts, generator=generators, **gen_kwargs)
            images = output.images
            
            if images and isinstance(images[0], list):
                images = [img for batch in images for img in batch]

        except Exception as e:
            print(f"[AquaLoRA] Generation failed: {e}")
            raise e
        finally:
            
            if hasattr(self, '_lora_saved') and self._lora_saved is not None:
                try:
                    from src.core.lora_utils import unmerge_lora
                    unmerge_lora(pipeline.unet, self._lora_saved)
                except Exception:
                    pass
                self._lora_saved = None
            
            if active_adapters:
                try:
                    pipeline.delete_adapters(active_adapters)
                except Exception:
                    try:
                        pipeline.unload_lora_weights()
                    except Exception:
                        pass
            
            if _tmp_lora is not None:
                try:
                    os.unlink(_tmp_lora.name)
                except Exception:
                    pass

        return images

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> Dict[str, Any]:
        
        _, simple_decode = self._ensure_deps()
        
        if secret is not None:
            if torch.is_tensor(secret):
                s_list = secret.cpu().numpy().flatten()
            else:
                s_list = np.array(secret).flatten()
            binary_str = ''.join(str(int(x)) for x in s_list)
        else:
            s_list = None
            binary_str = None
        
        decoder_path = f"{self.config['aqualora_model_path']}/msgdecoder.pt"
        
        if isinstance(image, list):
            images_to_process = image
        else:
            images_to_process = [image]
            
        acc, _, _ = simple_decode(
            len(s_list) if s_list is not None else 32, 
            decoder_path, 
            images_to_process, 
            msg_gt=binary_str
        )
        return [{'bit_acc': float(acc)}]
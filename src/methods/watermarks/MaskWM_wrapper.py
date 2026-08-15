
import os
import sys
import types
import importlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Union
from torchvision import transforms

from src.core import BaseWatermark
from src.core.registry import WATERMARKS
from src.core.paths import resolve_model_path

_ALIAS = "_maskwm_vendor_models"

def _maskwm_root() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'watermark_method', 'MaskWM'))

def _ensure_maskwm_imports():
    
    if f"{_ALIAS}.Mask_Model" in sys.modules:
        return

    if _ALIAS not in sys.modules:
        models_dir = os.path.join(_maskwm_root(), 'models')
        pkg = types.ModuleType(_ALIAS)
        pkg.__path__ = [models_dir]   
        pkg.__package__ = _ALIAS
        sys.modules[_ALIAS] = pkg

    importlib.import_module(f"{_ALIAS}.Mask_Model")

@WATERMARKS.register("MaskWM")
class MaskWMWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)

        self.device = torch.device(global_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.nbits = self.config.get('nbits', 32)
        self.img_size = self.config.get('img_size', 256)

        self.model_variant = self.config.get('model_variant', 'D')
        if self.model_variant not in ('D', 'ED'):
            raise ValueError(f"[MaskWM] model_variant must be 'D' or 'ED', got {self.model_variant}")
        self.ckpt_name = f"{self.model_variant}_{self.nbits}bits.pth"

        self.ckpt_dir = resolve_model_path(self.config.get('ckpt_dir', 'data/models/MaskWM'))

        default_jnd = 1.75 if self.model_variant == 'ED' else 1.3
        self.jnd_factor = float(self.config.get('jnd_factor', default_jnd))
        self.use_jnd = bool(self.config.get('use_jnd', True))
        self.jnd_blue = bool(self.config.get('jnd_blue', True))
        self.mask_threshold = float(self.config.get('mask_threshold', 0.5))

        self.model = None
        self._cached_msgs = None

        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.unnormalize = transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])

        self.num_inference_steps = self.config.get('num_inference_steps', 50)
        self.guidance_scale = self.config.get('guidance_scale', 7.5)

    def _ensure_model(self):
        if self.model is not None:
            return

        _ensure_maskwm_imports()
        WatermarkModel = sys.modules[f"{_ALIAS}.Mask_Model"].WatermarkModel
        import yaml

        wm_root = _maskwm_root()
        model_cfg_path = os.path.join(wm_root, 'configs', 'model', f"{self.model_variant}_{self.nbits}bits.yaml")
        if not os.path.exists(model_cfg_path):
            raise FileNotFoundError(f"[MaskWM] Model config not found: {model_cfg_path}")
        with open(model_cfg_path, 'r') as f:
            model_cfg = yaml.safe_load(f)

        print(f"   [MaskWM] Building WatermarkModel variant={self.model_variant} "
              )
        
        self.model = WatermarkModel(
            wm_enc_config=model_cfg['wm_enc_config'],
            wm_dec_config=model_cfg['wm_dec_config'],
            noise_layers="Identity()",
        )

        ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_name)
        if os.path.exists(ckpt_path):
            print(f"   [MaskWM] Loading weights from {ckpt_path}")
            sd = torch.load(ckpt_path, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd and len(sd) == 1:
                sd = sd['state_dict']
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing:
                print(f"   [MaskWM] ⚠️  Missing keys ({len(missing)}): {missing[:3]}...")
            if unexpected:
                print(f"   [MaskWM] ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
        else:
            print(f"   [MaskWM] ⚠️  Weights not found at {ckpt_path} — model is UNTRAINED")

        self.model.to(self.device).eval()

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        self._ensure_model()

        prompts = prompt if isinstance(prompt, list) else [prompt]
        seeds = kwargs.get('seed', 42)
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(len(prompts))]

        original = kwargs.get('original_image')
        if original is not None:
            clean = original if isinstance(original, list) else [original]
        else:
            clean = []
            for p, s in zip(prompts, seeds):
                gen = torch.Generator(self.device).manual_seed(s)
                out = pipeline(p, generator=gen,
                               num_inference_steps=self.num_inference_steps,
                               guidance_scale=self.guidance_scale,
                               height=kwargs.get('height', 512),
                               width=kwargs.get('width', 512))
                if hasattr(out, 'images'):
                    clean.append(out.images[0])
                else:
                    clean.append(out[0][0] if isinstance(out, tuple) else out[0])

        masks = kwargs.get('mask')
        if masks is not None and not isinstance(masks, list):
            masks = [masks] * len(clean)

        msgs = self._prep_msgs(len(prompts), secret)

        to_tensor = transforms.ToTensor()
        results = []
        for i, img in enumerate(clean):
            if img is None:
                results.append(None)
                continue
            if img.mode != 'RGB':
                img = img.convert('RGB')
            orig_size = img.size  

            t_full = to_tensor(img).unsqueeze(0).to(self.device)          
            t_full_norm = self.normalize(t_full)                          
            t_256_norm = F.interpolate(t_full_norm, size=(self.img_size, self.img_size),
                                        mode='bilinear', align_corners=False)
            m = msgs[i:i+1] if i < msgs.shape[0] else msgs[:1]

            mask_t = None
            if self.model_variant == 'ED' and masks is not None and masks[i] is not None:
                mask_img = masks[i].convert('L').resize((self.img_size, self.img_size), Image.NEAREST)
                mask_t = transforms.ToTensor()(mask_img).unsqueeze(0).to(self.device)  

            with torch.no_grad():
                wm_256_norm = self.model.encoder(
                    t_256_norm, m, mask=mask_t, use_jnd=self.use_jnd,
                    jnd_factor=self.jnd_factor, blue=self.jnd_blue,
                )
                residual_256 = wm_256_norm - t_256_norm
                residual_full = F.interpolate(residual_256, size=(orig_size[1], orig_size[0]),
                                               mode='bilinear', align_corners=False)
                wm_full_norm = (t_full_norm + residual_full).clamp(-1, 1)

            wm = self.unnormalize(wm_full_norm).clamp(0, 1)
            results.append(transforms.ToPILImage()(wm.squeeze(0).cpu()))

        self._cached_msgs = msgs.cpu()
        return results

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        self._ensure_model()
        images = image if isinstance(image, list) else [image]
        to_tensor = transforms.ToTensor()

        if secret is not None:
            gt = self._prep_msgs(1, secret)[0]
        elif self._cached_msgs is not None:
            gt = self._cached_msgs[0]
        else:
            gt = None

        results = []
        for img in images:
            if img is None:
                results.append({'bit_acc': 0.5})
                continue
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_rsz = img.resize((self.img_size, self.img_size), Image.BICUBIC)
            t = to_tensor(img_rsz).unsqueeze(0).to(self.device)
            t_norm = self.normalize(t)

            with torch.no_grad():
                decoded_message, mask_pred = self.model.decoder(t_norm)

            bits = (decoded_message[0] > self.mask_threshold).float()

            metrics = {'raw_bits': bits.cpu().tolist()}
            if gt is not None:
                n = min(len(bits), len(gt))
                metrics['bit_acc'] = float((bits[:n] == gt[:n].to(bits.device)).float().mean().item())
            else:
                metrics['bit_acc'] = 0.5
            results.append(metrics)

        return results

    def compute_aggregate_metrics(self, all_results: List[Dict[str, Any]]) -> Dict[str, float]:
        
        clean_results = [{k: v for k, v in r.items() if k != 'raw_mask_pred'} for r in all_results]
        from src.core.interfaces import BaseWatermark
        return BaseWatermark.compute_aggregate_metrics(self, clean_results)

    def _prep_msgs(self, bsz: int, secret: Any) -> torch.Tensor:
        
        if secret is None:
            return torch.randint(0, 2, (bsz, self.nbits), device=self.device).float()
        if torch.is_tensor(secret):
            msgs = secret.float().to(self.device)
        elif isinstance(secret, np.ndarray):
            msgs = torch.from_numpy(secret).float().to(self.device)
        elif isinstance(secret, list):
            msgs = torch.tensor(secret, dtype=torch.float32, device=self.device)
        else:
            return torch.randint(0, 2, (bsz, self.nbits), device=self.device).float()
        if msgs.dim() == 1:
            msgs = msgs.unsqueeze(0)
        if msgs.shape[0] < bsz:
            msgs = msgs.repeat((bsz + msgs.shape[0] - 1) // msgs.shape[0], 1)[:bsz]
        return msgs[:, :self.nbits]

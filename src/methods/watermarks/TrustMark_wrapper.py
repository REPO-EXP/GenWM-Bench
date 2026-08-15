"""
TrustMark (Adobe / ICCV 2025) wrapper for the Benchmark framework.

TrustMark is a *post-hoc* watermark: it takes an already-generated (or real)
RGB image and encodes an invisible bit-payload into it directly at pixel
level (256x256 residual, blended back at native resolution).  Decoding
recovers the payload (optionally protected with BCH error-correction) plus
a `detected` flag.

Vendor code lives at: src/watermark_method/trustmark/python/trustmark
Pretrained weights (encoder/decoder .ckpt + .yaml configs) are *not*
shipped with the repo — TrustMark downloads them on first use.  We redirect
that download location from the package-internal folder to
`data/models/TrustMark` (via a symlink) so the weights are kept alongside
every other method's checkpoints and persist across re-clones of the repo.
"""

import os
import sys
import shutil
import numpy as np
import torch
from PIL import Image
from typing import Any, Dict, List, Union

from src.core import BaseWatermark
from src.core.registry import WATERMARKS
from src.core.paths import resolve_model_path

_ENCODING_MAP = {
    'BCH_SUPER': 0,
    'BCH_5': 1,
    'BCH_4': 2,
    'BCH_3': 3,
}
_ENCODING_CAPACITY = {
    'BCH_SUPER': 40,
    'BCH_5': 61,
    'BCH_4': 68,
    'BCH_3': 75,
}

@WATERMARKS.register("TrustMark")
class TrustMarkWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        config = kwargs
        super().__init__(config, global_config)

        self.device = global_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        self.model_type = self.config.get('model_type', 'Q')
        if self.model_type not in ['B', 'C', 'P', 'Q']:
            raise ValueError(f"[TrustMark] Invalid model_type: {self.model_type} (expected B/C/P/Q)")

        enc_name = self.config.get('encoding_type', 'BCH_5')
        if enc_name not in _ENCODING_MAP:
            raise ValueError(f"[TrustMark] Unknown encoding_type: {enc_name} "
                              f"(expected one of {list(_ENCODING_MAP.keys())})")
        self.encoding_name = enc_name
        self.encoding_type = _ENCODING_MAP[enc_name]

        self.use_ecc = bool(self.config.get('use_ecc', True))

        self.model_secret_len = 100
        
        self.capacity = _ENCODING_CAPACITY[enc_name] if self.use_ecc else self.model_secret_len

        self.wm_strength = float(self.config.get('wm_strength', 1.0))
        self.wm_merge = self.config.get('wm_merge', 'bilinear')
        self.concentrate_wm_region = float(self.config.get('concentrate_wm_region', 1.0))

        self.detect_first = bool(self.config.get('detect_first', False))
        self.rotation = bool(self.config.get('rotation', False))
        self.load_bbox_detector = bool(self.config.get('load_bbox_detector', self.detect_first))
        self.load_remover = bool(self.config.get('load_remover', False))

        self.model_dir = self.config.get('model_dir', 'data/models/TrustMark')

        self.tm = None
        self._cached_secret_bits = None  
        self._cached_raw_bits = None     

    def _ensure_model_dir_link(self, tm_module_file: str):
        """
        TrustMark  models/ ( trustmark.py )
        
        -  data/models/TrustMark  models/
        - wfs Python import  TrustMark
           check_and_download models/ 
        """
        pkg_models_dir = os.path.join(os.path.dirname(os.path.abspath(tm_module_file)), 'models')
        target_dir = resolve_model_path(self.model_dir)

        os.makedirs(target_dir, exist_ok=True)

        if os.path.islink(pkg_models_dir):
            real_target = os.readlink(pkg_models_dir)
            if real_target == target_dir:
                return
            os.unlink(pkg_models_dir)

        if not os.path.exists(pkg_models_dir):
            try:
                os.symlink(target_dir, pkg_models_dir)
                return
            except OSError:
                pass  

        os.makedirs(pkg_models_dir, exist_ok=True)

        if os.path.isdir(target_dir):
            for fname in os.listdir(target_dir):
                src_f = os.path.join(target_dir, fname)
                dst_f = os.path.join(pkg_models_dir, fname)
                if os.path.isfile(src_f) and not os.path.exists(dst_f):
                    try:
                        shutil.copy2(src_f, dst_f)
                    except OSError as e:
                        print(f"[TrustMark WARNING] Failed to copy {fname}: {e}")

    def _load_deps(self):
        if self.tm is not None:
            return

        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        trustmark_py_dir = os.path.join(project_root, "src", "watermark_method", "trustmark", "python")
        if trustmark_py_dir not in sys.path:
            sys.path.insert(0, trustmark_py_dir)

        try:
            from trustmark import TrustMark
            import trustmark.trustmark as _tm_module
        except ImportError as e:
            raise ImportError(f"[TrustMark] Failed to import vendor package from {trustmark_py_dir}: {e}")

        self._ensure_model_dir_link(_tm_module.__file__)

        print(f"[TrustMark] Loading model_type={self.model_type} encoding={self.encoding_name} "
              f"use_ECC={self.use_ecc} device={self.device} (weights dir: {resolve_model_path(self.model_dir)}) ...")
        try:
            self.tm = TrustMark(
                verbose=True,
                use_ECC=self.use_ecc,
                secret_len=self.model_secret_len,
                device=self.device,
                model_type=self.model_type,
                encoding_type=self.encoding_type,
                concentrate_wm_region=self.concentrate_wm_region,
                loadRemover=self.load_remover,
                loadBBoxDetector=self.load_bbox_detector,
            )
        except Exception as e:
            print(f"[TrustMark ERROR] Model init/download failed: {e}")
            raise e
        print("[TrustMark] Model loaded successfully.")

    def _to_bitstring(self, bits: np.ndarray, length: int) -> str:
        bits = bits.astype(int).flatten().tolist()
        if len(bits) == 0:
            bits = [0] * length
        if len(bits) < length:
            reps = length // len(bits) + 1
            bits = (bits * reps)[:length]
        else:
            bits = bits[:length]
        return ''.join(str(int(b)) for b in bits)

    def _prep_secret_array(self, secret: Any, batch_size: int) -> np.ndarray:
        """ secret  (batch_size, capacity)  0/1 numpy """
        if secret is None:
            arr = np.random.randint(0, 2, size=(1, self.capacity))
        elif torch.is_tensor(secret):
            arr = secret.detach().cpu().numpy()
        elif isinstance(secret, np.ndarray):
            arr = secret
        elif isinstance(secret, list):
            arr = np.array(secret)
        else:
            arr = np.random.randint(0, 2, size=(1, self.capacity))

        if arr.ndim == 1:
            arr = arr[None, :]

        if arr.shape[0] == 1 and batch_size > 1:
            arr = np.repeat(arr, batch_size, axis=0)
        elif arr.shape[0] != batch_size:
            reps = batch_size // arr.shape[0] + 1
            arr = np.tile(arr, (reps, 1))[:batch_size]

        return arr

    def embed(self, pipeline, prompt: Union[str, List[str]], secret: Any, **kwargs) -> List[Image.Image]:
        self._load_deps()

        prompts = prompt if isinstance(prompt, list) else [prompt]

        raw_seed = kwargs.get('seed', 42)
        if isinstance(raw_seed, list):
            seeds = [int(s) for s in raw_seed]
        else:
            seeds = [int(raw_seed) + i for i in range(len(prompts))]

        original = kwargs.get('original_image')
        if original is not None:
            clean_images = original if isinstance(original, list) else [original]
        elif pipeline is not None:
            gen_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ['seed', 'original_image', 'output_dir', 'global_config']}
            if 'num_inference_steps' not in gen_kwargs:
                gen_kwargs['num_inference_steps'] = 50
            if 'guidance_scale' not in gen_kwargs:
                gen_kwargs['guidance_scale'] = 7.5

            min_len = min(len(prompts), len(seeds))
            gen_prompts = prompts[:min_len]
            generators = [torch.Generator(self.device).manual_seed(s) for s in seeds[:min_len]]

            try:
                out = pipeline(gen_prompts, generator=generators, **gen_kwargs)
                clean_images = out.images if hasattr(out, 'images') else out[0]
            except Exception as e:
                print(f"[TrustMark ERROR] Base image generation failed: {e}")
                import traceback; traceback.print_exc()
                return []
        else:
            raise ValueError("[TrustMark] embed()  pipeline  original_image ")

        if len(clean_images) == 0:
            return []

        out_dir = kwargs.get('output_dir', '')
        if out_dir:
            cd = os.path.join(out_dir, 'clean_images')
            os.makedirs(cd, exist_ok=True)
            for ci, cimg in enumerate(clean_images):
                if hasattr(cimg, 'save'):
                    cimg.save(os.path.join(cd, f"sample_{ci}.png"))

        batch_size = len(clean_images)

        secret_arr = self._prep_secret_array(secret, batch_size)

        output_images = []
        raw_bits_cache = []  
        for i, img in enumerate(clean_images):
            if img is None:
                output_images.append(None)
                raw_bits_cache.append(None)
                continue
            try:
                rgb = img.convert('RGB') if img.mode != 'RGB' else img
                bitstring = self._to_bitstring(secret_arr[i], self.capacity)
                
                if self.use_ecc and self.tm.ecc is not None:
                    raw_packet = self.tm.ecc.encode_binary([bitstring])[0].astype(int)
                else:
                    raw_packet = np.array([int(c) for c in bitstring], dtype=int)
                raw_bits_cache.append(raw_packet)
                stego = self.tm.encode(
                    rgb, bitstring, MODE='binary',
                    WM_STRENGTH=self.wm_strength, WM_MERGE=self.wm_merge,
                )
                output_images.append(stego)
            except Exception as e:
                print(f"[TrustMark ERROR] Embed failed on sample {i}: {e}")
                import traceback; traceback.print_exc()
                output_images.append(img)
                raw_bits_cache.append(None)

        self._cached_secret_bits = secret_arr
        self._cached_raw_bits = np.array([r for r in raw_bits_cache if r is not None]) if any(r is not None for r in raw_bits_cache) else None
        return output_images

    def extract(self, image: Union[Image.Image, List[Image.Image]], secret: Any = None, **kwargs) -> List[Dict[str, Any]]:
        self._load_deps()
        images = image if isinstance(image, list) else [image]

        if secret is not None:
            target_arr = self._prep_secret_array(secret, len(images))
        elif self._cached_secret_bits is not None:
            cached = self._cached_secret_bits
            if cached.shape[0] == 1 and len(images) > 1:
                target_arr = np.repeat(cached, len(images), axis=0)
            elif cached.shape[0] != len(images):
                reps = len(images) // cached.shape[0] + 1
                target_arr = np.tile(cached, (reps, 1))[:len(images)]
            else:
                target_arr = cached
        else:
            target_arr = None

        if self._cached_raw_bits is not None and self.use_ecc:
            raw_target = self._cached_raw_bits
            if raw_target.shape[0] == 1 and len(images) > 1:
                raw_target = np.repeat(raw_target, len(images), axis=0)
            elif raw_target.shape[0] != len(images):
                reps = len(images) // raw_target.shape[0] + 1
                raw_target = np.tile(raw_target, (reps, 1))[:len(images)]
        else:
            raw_target = None

        results = []
        for i, img in enumerate(images):
            if img is None:
                results.append({'bit_acc': 0.0, 'detected': False, 'error': 'None image'})
                continue
            try:
                rgb = img.convert('RGB') if img.mode != 'RGB' else img
                secret_pred, detected, schema = self.tm.decode(
                    rgb, MODE='binary', DETECTFIRST=self.detect_first, ROTATION=self.rotation
                )

                pred_bits = [int(c) for c in secret_pred] if secret_pred else []
                metrics = {
                    'raw_bits': pred_bits,
                    'detected': bool(detected),
                    'schema': int(schema),
                }

                if target_arr is not None:
                    tgt = target_arr[i].astype(int).flatten().tolist()
                    if len(pred_bits) > 0:
                        
                        min_len = min(len(pred_bits), len(tgt))
                        acc = float(np.mean(np.array(pred_bits[:min_len]) == np.array(tgt[:min_len])))
                    elif raw_target is not None:
                        
                        from torchvision import transforms as T
                        stego_tensor = T.ToTensor()(
                            rgb.resize((self.tm.model_resolution_dec, self.tm.model_resolution_dec),
                                       Image.BILINEAR)
                        ).unsqueeze(0).to(self.tm.decoder.device) * 2.0 - 1.0
                        with torch.no_grad():
                            raw_pred = (self.tm.decoder.decoder(stego_tensor) > 0).cpu().numpy()[0].astype(int)
                        raw_tgt = raw_target[i].astype(int).flatten()
                        compare_len = min(len(raw_pred), len(raw_tgt))
                        acc = float(np.mean(raw_pred[:compare_len] == raw_tgt[:compare_len]))
                        metrics['raw_bits'] = raw_pred.tolist()
                    else:
                        acc = 0.0
                    metrics['bit_acc'] = acc
                else:
                    metrics['bit_acc'] = 0.0

                results.append(metrics)
            except Exception as e:
                print(f"[TrustMark ERROR] Extract failed on sample {i}: {e}")
                import traceback; traceback.print_exc()
                results.append({'bit_acc': 0.0, 'detected': False, 'error': str(e)})

        return results

    def compute_aggregate_metrics(self, all_sample_results: List[Dict[str, Any]]) -> Dict[str, float]:
        metrics = super().compute_aggregate_metrics(all_sample_results)
        if all_sample_results and 'detected' in all_sample_results[0]:
            metrics['detect_rate'] = float(np.mean(
                [1.0 if r.get('detected') else 0.0 for r in all_sample_results]
            ))
        return metrics

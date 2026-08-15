import os
import pickle
import torch
import numpy as np
import traceback
from typing import Any, Dict, List
from src.core import BaseWatermark
from src.core.registry import WATERMARKS
from src.core.paths import resolve_model_path

@WATERMARKS.register("PRC")
class PRCWatermark(BaseWatermark):
    def __init__(self, **kwargs):
        global_config = kwargs.pop('global_config', {})
        super().__init__(kwargs, global_config)
        self.device = torch.device(global_config.get('device', 'cuda'))
        self.n = 16384
        self.model_id = resolve_model_path(self.config.get('model_path', 'stabilityai/stable-diffusion-2-1-base'))
        self.secret_length = 512 
        output_dir = self.config.get('output_dir', 'outputs')
        self.persistence_path = os.path.join(output_dir, 'prc_state.pkl')
        
        from src.watermark_method.PRC.src.prc import KeyGen
        
        if os.path.exists(self.persistence_path):
            print(f">>> [PRC] Loading state from {self.persistence_path}")
            with open(self.persistence_path, 'rb') as f:
                state = pickle.load(f)
                self.enc_key = state['enc_key']
                self.dec_key = state['dec_key']
                self.clean_codewords_dict = state.get('clean_codewords_dict', {})
        else:
            print(">>> [PRC] Generating new keys...")
            self.enc_key, self.dec_key = KeyGen(self.n, message_length=self.secret_length)
            self.clean_codewords_dict = {} 
            
        self.pipe = None

    def _save_state(self):
        """"""
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        with open(self.persistence_path, 'wb') as f:
            pickle.dump({
                'enc_key': self.enc_key,
                'dec_key': self.dec_key,
                'clean_codewords_dict': self.clean_codewords_dict 
            }, f)

    def _load_pipe(self):
        if self.pipe is not None: return
        from src.watermark_method.PRC.inversion import stable_diffusion_pipe
        self.pipe = stable_diffusion_pipe(solver_order=1, model_id=self.model_id)
        if hasattr(self.pipe, "safety_checker") and self.pipe.safety_checker is not None:
            self.pipe.safety_checker = None
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def embed(self, pipeline, prompt, secret=None, **kwargs) -> List[Any]:
        try:
            self._load_pipe()
            from src.watermark_method.PRC.src.prc import Encode
            import src.watermark_method.PRC.src.pseudogaussians as prc_gaussians
            from src.watermark_method.PRC.src.optim_utils import set_random_seed

            seeds = kwargs.get('seed', 42)
            if not isinstance(seeds, list): seeds = [seeds]
            
            latents = []
            for i in range(len(seeds)):
                current_seed = int(seeds[i])
                set_random_seed(current_seed)
                sig, clean_cw, _ = Encode(self.enc_key, message=secret)
                
                self.clean_codewords_dict[current_seed] = clean_cw
                
                latent = prc_gaussians.sample(sig).reshape(1, 4, 64, 64)
                latents.append(latent.to(self.device, dtype=self.pipe.unet.dtype))
            
            self._save_state() 
            
            out = self.pipe(prompt, latents=torch.cat(latents), num_inference_steps=50)
            return out.images if not isinstance(out, tuple) else out[0].images
        except Exception:
            traceback.print_exc(); return []

    def extract(self, image, secret=None, **kwargs) -> List[Dict[str, Any]]:
        try:
            self._load_pipe()
            from src.watermark_method.PRC.src.prc import Detect, Decode
            import src.watermark_method.PRC.src.pseudogaussians as prc_gaussians
            from src.watermark_method.PRC.src.optim_utils import transform_img, set_random_seed

            images = image if isinstance(image, list) else [image]
            seeds = kwargs.get('seed', 42)
            if not isinstance(seeds, list): seeds = [seeds]

            imgs_t = torch.cat([transform_img(img.convert("RGB").resize((512, 512))).unsqueeze(0) for img in images]).to(self.device)
            latents = self.pipe.get_image_latents(imgs_t, sample=False)
            inv_l = self.pipe.forward_diffusion(latents=latents, text_embeddings=self.pipe.get_text_embedding(['']*len(images)), guidance_scale=1.0, num_inference_steps=50)
            
            results = []
            for i in range(len(images)):
                current_seed = int(seeds[i])
                set_random_seed(current_seed)
                post = prc_gaussians.recover_posteriors(inv_l[i:i+1].detach().to(torch.float64).flatten().cpu(), variances=1.5)
                
                raw_extracted_bits = (post.numpy() < 0).astype(np.int64).flatten()
                raw_bit_acc = 0.0
                
                target_clean_cw = self.clean_codewords_dict.get(current_seed)
                if target_clean_cw is not None:
                    target_np = np.array(target_clean_cw).flatten()
                    
                    if np.min(target_np) < 0:
                        target_bits = (target_np > 0).astype(np.int64)
                    else:
                        target_bits = target_np.astype(np.int64)

                    min_cw_l = min(len(raw_extracted_bits), len(target_bits))
                    if min_cw_l > 0:
                        
                        match_normal = float((raw_extracted_bits[:min_cw_l] == target_bits[:min_cw_l]).mean())
                        match_invert = float((raw_extracted_bits[:min_cw_l] == (1 - target_bits[:min_cw_l])).mean())
                        raw_bit_acc = max(match_normal, match_invert)
                else:
                    print(f"Warning: No clean_codeword found for seed {current_seed}! Did you run embed first?")

                is_det = Detect(self.dec_key, post)
                decoded_payload = Decode(self.dec_key, post) 
                
                decoded_msg_acc = 0.0
                target_msg = secret.detach().cpu().numpy().flatten().astype(np.int64) if torch.is_tensor(secret) else np.array(secret).flatten().astype(np.int64)
                extracted_msg_np = np.array([])
                
                if decoded_payload is not None:
                    t_len, g_len = len(self.dec_key[5]), self.dec_key[6]
                    raw_extracted = decoded_payload[t_len + g_len : t_len + g_len + len(target_msg)]
                    
                    extracted_msg_np = np.array(raw_extracted).flatten().astype(np.int64)
                    
                    min_l = min(len(extracted_msg_np), len(target_msg))
                    if min_l > 0:
                        decoded_msg_acc = float((extracted_msg_np[:min_l] == target_msg[:min_l]).mean())

                results.append({
                    'bit_acc': raw_bit_acc,
                    'decoded_acc': decoded_msg_acc,
                    'is_watermarked': bool(is_det),
                    'raw_bits': extracted_msg_np.tolist() if extracted_msg_np.size > 0 else []
                })
            return results
        except Exception:
            traceback.print_exc(); return [{'bit_acc': 0.0, 'decoded_acc': 0.0, 'is_watermarked': False}] * len(images)
            
    def compute_aggregate_metrics(self, all_sample_results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not all_sample_results: return {}
        metrics = {}
        for key in ['bit_acc', 'decoded_acc', 'is_watermarked']:
            values = [res[key] for res in all_sample_results if isinstance(res.get(key), (int, float, bool))]
            if values:
                metrics[f"avg_{key}"] = float(np.mean(values))
        
        if 'bit_acc' in all_sample_results[0]:
            bit_accs = [res['bit_acc'] for res in all_sample_results]
            bit_accs_decoded = [res['decoded_acc'] for res in all_sample_results]
            metrics['TPR@1e-2_decoded'] = self._calc_bit_based_tpr(bit_accs_decoded, self.n, 1e-2)
            metrics['TPR@1e-6_decoded'] = self._calc_bit_based_tpr(bit_accs_decoded, self.n, 1e-6)

            metrics['TPR@1e-2ori'] = self._calc_bit_based_tpr(bit_accs, self.n, 1e-2)
            metrics['TPR@1e-6ori'] = self._calc_bit_based_tpr(bit_accs, self.n, 1e-6)
        return metrics
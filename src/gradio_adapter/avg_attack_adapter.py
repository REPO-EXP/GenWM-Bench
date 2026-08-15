import os
import torch
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as TF

class AVGAttackGradioAdapter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.watermark = None
        self.current_wm_name = None
        self.current_model_path = None
        self.wm_config = None
        self.CONFIG_ROOT = "configs"
        self.AVG_OUTPUT_ROOT = "outputs/avg_attack_eval"

    def update_dir_by_wm(self, wm_name, ds_name):
        wm_clean = wm_name.replace('.yaml', '')
        ds_clean = ds_name.replace('.yaml', '')
        return os.path.join(self.AVG_OUTPUT_ROOT, wm_clean, f"{wm_clean}_{ds_clean}")

    def _load_yaml(self, path):
        if not os.path.exists(path): return {}
        with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

    def _generate_secret(self, wm_config):
        if 'fixed_secret' in wm_config: 
            return torch.tensor(wm_config['fixed_secret'], device=self.device).float()
        state = torch.get_rng_state()
        torch.manual_seed(42) 
        s_cfg = wm_config.get('secret_config', {})
        length = s_cfg.get('length', 48)
        secret = torch.randint(0, 2, (length,), device=self.device).float()
        torch.set_rng_state(state) 
        return secret

    def calculate_psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0: return 100.0
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    def load_components(self, wm_name, model_name):
        from src.core.registry import MODELS, WATERMARKS
        from src.core.loader import setup_env
        setup_env()
        wm_cfg_path = os.path.join(self.CONFIG_ROOT, "methods/watermarks", wm_name)
        model_cfg_path = os.path.join(self.CONFIG_ROOT, "models", model_name)
        
        self.wm_config = self._load_yaml(wm_cfg_path)
        model_file_cfg = self._load_yaml(model_cfg_path)
        target_model_path = model_file_cfg.get('model_id', "runwayml/stable-diffusion-v1-5")

        if self.current_model_path != target_model_path:
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
            self.model = MODELS.build({"name": "StableDiffusion", "model_id": target_model_path, "device": self.device, "dtype": torch.float16})
            self.current_model_path = target_model_path

        if self.current_wm_name != wm_name or self.watermark is None:
            self.wm_config['global_config'] = {"device": self.device}
            self.watermark = WATERMARKS.build(self.wm_config)
            self.current_wm_name = wm_name
        return True

    def run_triple_generation(self, wm_cfg_name, ds_cfg_name, model_cfg_name, seed, max_samples):
        self.load_components(wm_cfg_name, model_cfg_name)
        from src.core.registry import DATASETS
        ds_cfg = self._load_yaml(os.path.join(self.CONFIG_ROOT, "datasets", ds_cfg_name))
        dataset = DATASETS.build(ds_cfg)

        exp_dir = self.update_dir_by_wm(wm_cfg_name, ds_cfg_name)
        paths = {k: os.path.join(exp_dir, k) for k in ["clean", "watermarked", "random_distractor"]}
        for p in paths.values(): os.makedirs(p, exist_ok=True)

        yield f"🚀 [AVG-Prep] Initializing generation...\n📂 Directory: {exp_dir}\n"
        global_secret = self._generate_secret(self.wm_config)
        steps = self.wm_config.get('inference_steps', 50)

        skipped = 0
        for i in range(int(max_samples)):
            c_seed = int(seed) + i
            fn = f"sample_{c_seed}.png"
            cp, wp, rp = [os.path.join(paths[k], fn) for k in ["clean", "watermarked", "random_distractor"]]
            
            if all(os.path.exists(p) for p in [cp, wp, rp]):
                skipped += 1; continue
            if skipped > 0:
                yield f"⏩ Skipped {skipped} existing triplets\n"; skipped = 0

            try:
                item = dataset[i]
                curr_p = item.get('prompt', item.get('caption', "")) if isinstance(item, dict) else str(item[0] if isinstance(item, (list, tuple)) else item)
            except: break

            with torch.no_grad():
                yield f"🎨 [{i+1}/{int(max_samples)}] Generating: {fn}\n"
                
                gen_c = torch.Generator(self.device).manual_seed(c_seed)
                clean_img = self.model.pipe(curr_p, num_inference_steps=steps, generator=gen_c).images[0]
                clean_img.save(cp)

                wm_img = self.watermark.embed(pipeline=self.model.pipe, prompt=curr_p, secret=global_secret, seed=c_seed, original_image=clean_img, num_inference_steps=steps)
                if isinstance(wm_img, list): wm_img = wm_img[0]
                wm_img.save(wp)

                gen_r = torch.Generator(self.device).manual_seed(c_seed + 10000)
                rand_img = self.model.pipe(curr_p, num_inference_steps=steps, generator=gen_r).images[0]
                rand_img.save(rp)
                yield f"   ✅ Clean/WM/Random versions saved\n"
        yield f"🏁 Generation task completed.\n"

    def run_avg_attack_test(self, wm_cfg_name, model_cfg_name, mode, box_type, exp_dir, n_ext, n_tst):
        self.load_components(wm_cfg_name, model_cfg_name)
        global_secret = self._generate_secret(self.wm_config)
        
        yield f"\n🛠️ [AVG-Attack] Mode: {mode.upper()} | Scenario: {box_type}\n", None
        
        avg_pattern = None
        
        if mode.lower() != "detect":
            yield f"🧪 Extracting average pattern ({box_type}, N={n_ext})...\n", None
            res_list = []
            wm_dir = os.path.join(exp_dir, "watermarked")
            wm_files = sorted([f for f in os.listdir(wm_dir) if f.endswith('.png')])[:int(n_ext)]
            
            ref_dirname = "clean" if box_type == "Gray-box" else "random_distractor"
            ref_dir = os.path.join(exp_dir, ref_dirname)

            for fn in wm_files:
                wm_p, ref_p = os.path.join(wm_dir, fn), os.path.join(ref_dir, fn)
                if os.path.exists(wm_p) and os.path.exists(ref_p):
                    w_t = TF.to_tensor(Image.open(wm_p)).to(self.device)
                    r_t = TF.to_tensor(Image.open(ref_p)).to(self.device)
                    
                    res_list.append(w_t - r_t)
            
            if not res_list:
                yield f"❌ Error: Reference samples not found ({ref_dirname})\n", None; return
            
            avg_pattern = torch.stack(res_list).mean(dim=0)
            yield f"✅ Pattern extraction completed.\n", None

        wm_ref_dir = os.path.join(exp_dir, "watermarked")
        all_files = sorted([f for f in os.listdir(wm_ref_dir) if f.endswith('.png')])
        
        test_files = all_files[-int(n_tst):] if len(all_files) >= int(n_tst) else all_files
        
        psnr_list = []
        all_sample_metrics = []
        
        for fn in test_files:
            try:
                seed = int(fn.split('_')[1].split('.')[0])
                img_wm_t = TF.to_tensor(Image.open(os.path.join(wm_ref_dir, fn))).to(self.device)
                
                if mode.lower() == "forgery":
                    ref_type = "random_distractor" if box_type == "Black-box" else "clean"
                    src_p = os.path.join(exp_dir, ref_type, fn)
                    if not os.path.exists(src_p): continue
                    base_t = TF.to_tensor(Image.open(src_p)).to(self.device)
                    atk_t = torch.clamp(base_t + avg_pattern, 0, 1)
                    ref_for_psnr = base_t
                elif mode.lower() == "removal":
                    atk_t = torch.clamp(img_wm_t - avg_pattern, 0, 1)
                    ref_for_psnr = img_wm_t
                else: 
                    
                    atk_t, ref_for_psnr = img_wm_t, img_wm_t 
                
                psnr = self.calculate_psnr(ref_for_psnr, atk_t)
                psnr_list.append(psnr)
                
                res = self.watermark.extract(TF.to_pil_image(atk_t.cpu()), secret=global_secret, seed=seed)
                
                if isinstance(res, list):
                    all_sample_metrics.extend(res)
                    res_dict = res[0] if res else {}
                else:
                    all_sample_metrics.append(res)
                    res_dict = res
                
                display_acc = res_dict.get('bit_acc', res_dict.get('decoded_acc', 0.0))
                if display_acc == 0.0 and "raw_bits" in res_dict and res_dict["raw_bits"] is not None:
                    pred = torch.tensor(res_dict["raw_bits"]).to(self.device).float()
                    min_l = min(len(pred), len(global_secret))
                    display_acc = (pred[:min_l] == global_secret[:min_l]).float().mean().item()
                
                yield f"📄 {fn} | PSNR: {psnr:.2f} | Acc: {display_acc:.2%}\n", None
            except Exception as e:
                yield f"⚠️ Error processing {fn}: {str(e)}\n", None

        final_metrics = {}
        if all_sample_metrics and hasattr(self.watermark, 'compute_aggregate_metrics'):
            final_metrics = self.watermark.compute_aggregate_metrics(all_sample_metrics)

        df_rows = [{"Metric": "Average PSNR", "Value": f"{np.mean(psnr_list):.2f} dB" if psnr_list else "0.00 dB"}]
        
        for k, v in final_metrics.items():
            if isinstance(v, float):

                val_str = f"{v:.2%}" if "acc" in k.lower() or "tpr" in k.lower() else f"{v:.4f}"
            else:
                val_str = str(v)
            df_rows.append({"Metric": f"WM: {k}", "Value": val_str})

        df = pd.DataFrame(df_rows)
        yield f"\n📊 Evaluation completed.\n", df
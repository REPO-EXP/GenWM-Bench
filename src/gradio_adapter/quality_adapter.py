
import os
import torch
import gc
import yaml
import sys
import pandas as pd
import numpy as np
import time
import json
from PIL import Image, ImageChops
from torch.utils.data import DataLoader 
from torchvision.transforms import functional as TF

sys.path.append(os.path.abspath("."))

try:
    from src.core.registry import WATERMARKS, MODELS, METRICS
    from src.core.config_parser import ConfigManager
    from src.core.loader import setup_env
    setup_env()
except ImportError as e:
    print(f"Warning: src.core modules import failed: {e}")

METRIC_BEHAVIORS = {
    'BasicQualityMetrics': 'fidelity_group', 
    'SFIDMetric':          'fidelity_group', 
    'NIQEMetric':          'no_ref',
    'PIQEMetric':          'no_ref',
    'CLIPScoreMetric':     'alignment',
    'DreamSimMetric':      'fidelity_group',
    'FIDMetric':           'distribution',
}

class QualityGradioAdapter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.watermark = None
        self.current_wm_name = None
        self.current_model_path = None
        self.CONFIG_ROOT = "configs"
        self.dataset = None

    def _get_model_config(self, model_path):
        cfg = {
            "name": "StableDiffusion", 
            "model_id": model_path,
            "device": self.device
        }
        if "v1-5" in model_path or "v2-1" in model_path: 
            cfg["dtype"] = "fp16"
        return cfg

    def _load_yaml(self, path):
        if not os.path.exists(path): return {}
        return ConfigManager.load_yaml(path)

    def _to_tensor(self, img):
        if img is None: return None
        return TF.to_tensor(img).unsqueeze(0).to(self.device)

    def load_components(self, wm_name, model_name):
        status_msg = []
        
        wm_config_path = os.path.join(self.CONFIG_ROOT, "methods/watermarks", wm_name)
        try:
            wm_cfg = self._load_yaml(wm_config_path)
        except Exception as e:
            return False, f"Config Error: {e}"

        model_config_path = os.path.join(self.CONFIG_ROOT, "models", model_name)
        try:
            model_file_cfg = self._load_yaml(model_config_path)
            target_model_path = model_file_cfg.get('model_id', "runwayml/stable-diffusion-v1-5")
        except:
            target_model_path = "runwayml/stable-diffusion-v1-5" 

        if self.current_model_path != target_model_path:
            print(f">>> [Adapter] Auto-Loading Model: {target_model_path}...")
            try:
                if self.model is not None:
                    self.unload_components(keep_watermark=False)
                
                model_cfg = self._get_model_config(target_model_path)
                self.model = MODELS.build(model_cfg)
                self.current_model_path = target_model_path
                status_msg.append(f"Model: {os.path.basename(target_model_path)}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                return False, f"Model Load Error: {e}"

        if self.current_wm_name != wm_name or self.watermark is None:
            print(f">>> [Adapter] Loading Watermark: {wm_name}...")
            try:
                wm_cfg['model_path'] = target_model_path
                wm_cfg['global_config'] = {
                    "device": self.device,
                    "model_config": self._get_model_config(target_model_path)
                }
                self.watermark = WATERMARKS.build(wm_cfg)
                self.current_wm_name = wm_name
                status_msg.append(f"Algorithm: {wm_name}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.watermark = None 
                return False, f"WM Init Error: {e}"
        
        return True, " | ".join(status_msg)

    def unload_components(self, keep_watermark=False):
        print(">>> [Adapter] Releasing VRAM...")
        if self.model is not None:
            del self.model
            self.model = None
            self.current_model_path = None

        if not keep_watermark and self.watermark is not None:
            del self.watermark
            self.watermark = None
            self.current_wm_name = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "Memory Cleared"

    def load_template_params(self, exp_name):
        path = os.path.join(self.CONFIG_ROOT, "experiments", exp_name)
        if not os.path.exists(path): return []
        try:
            with open(path, 'r') as f: cfg = yaml.safe_load(f)
            defaults = cfg.get('defaults', {})
            wm_raw = (defaults.get('method') or defaults.get('watermark_config') or defaults.get('watermark') or cfg.get('method') or 'ZoDiac.yaml')
            ds_raw = (defaults.get('dataset') or defaults.get('dataset_config') or cfg.get('dataset') or 'coco.yaml')
            model_raw = (defaults.get('model') or defaults.get('model_config') or cfg.get('model') or 'sd_v1_5.yaml')
            return [
                os.path.basename(str(wm_raw)),
                os.path.basename(str(ds_raw)),
                os.path.basename(str(model_raw)),
                cfg.get('output_dir', 'outputs/debug'),
                cfg.get('seed', 42),
                cfg.get('max_samples', 5),
                cfg.get('batch_size', 1)
            ]
        except: return []

    def run_single_preview(self, wm_name, model_name, prompt, seed):
        success, msg = self.load_components(wm_name, model_name)
        if not success: return None, None, None, pd.DataFrame(), msg
        
        current_seed = int(seed)
        try:
            print(">>> Generating Clean Reference...")
            generator = torch.Generator(device=self.device).manual_seed(current_seed)
            with torch.no_grad():
                clean_img = self.model.pipe(prompt, generator=generator).images[0]

            print(">>> Embedding Watermark...")
            secret_len = 32
            if hasattr(self.watermark, 'config'):
                 secret_len = self.watermark.config.get('secret_config', {}).get('length', 32)
            
            secret = torch.randint(2, (int(secret_len),), device=self.device).float()

            with torch.set_grad_enabled(True):
                wm_results = self.watermark.embed(
                    pipeline=self.model.pipe, prompt=[prompt], secret=secret,
                    seed=[current_seed], original_image=clean_img 
                )
            
            if not wm_results or wm_results[0] is None:
                return clean_img, None, None, pd.DataFrame(), "Error: Embed returned None"

            wm_img = wm_results[0]
            diff_img = ImageChops.difference(clean_img, wm_img).point(lambda p: p*10).convert('L')
            
            metrics_data = []
            try:
                t_c = self._to_tensor(clean_img)
                t_w = self._to_tensor(wm_img)
                metric_calc = METRICS.build({"name": "BasicQualityMetrics", "device": self.device})
                raw_res = metric_calc.calculate(img_gen_clean=t_c, img_gen_wm=t_w, img_orig=t_c, img_wm=t_w)
                for k, v in raw_res.items():
                    if k.lower() in ['psnr', 'ssim', 'mssim', 'lpips']:
                        metrics_data.append({"Metric": k.upper(), "Value": round(float(v), 4)})
            except Exception as e:
                print(f"Warning: Full metrics calculation failed ({e})")
                metrics_data.append({"Metric": "Error", "Value": str(e)})
            
            df = pd.DataFrame(metrics_data)
            return clean_img, wm_img, diff_img, df, f"Success: {msg}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            return None, None, None, pd.DataFrame(), f"Runtime Error: {e}"

    def _parse_dataset_item(self, item, idx):
        gt_img = None
        prompt = ""
        filename = f"{idx:05d}" 

        if isinstance(item, dict):
            prompt = item.get('prompt', item.get('caption', ""))
            if isinstance(prompt, list): prompt = prompt[0]
            
            if 'filename' in item: filename = str(item['filename'])
            elif 'id' in item: filename = str(item['id'])
            filename = os.path.splitext(os.path.basename(filename))[0]

            keys_to_check = ['gt_image', 'image', 'img', 'target', 'jpg']
            for k in keys_to_check:
                if k in item and item[k] is not None:
                    val = item[k]
                    if isinstance(val, torch.Tensor): gt_img = TF.to_pil_image(val)
                    elif isinstance(val, Image.Image): gt_img = val
                    break
        else:
            prompt = str(item)
        return prompt, filename, gt_img

    def run_batch_generation(self, wm_name, ds_name, model_name, out_root, start_seed, max_samples, batch_size=1):
        success, msg = self.load_components(wm_name, model_name)
        if not success:
            yield f"Error loading components: {msg}"
            return

        wm_clean_name = wm_name.replace('.yaml', '')
        ds_clean_name = ds_name.replace('.yaml', '')
        exp_folder = f"{wm_clean_name}_{ds_clean_name}"
        full_out_dir = os.path.join(out_root, exp_folder)
        
        path_clean = os.path.join(full_out_dir, "clean")
        path_wm = os.path.join(full_out_dir, "watermarked")
        path_gt = os.path.join(full_out_dir, "ground_truth")
        for d in [path_clean, path_wm, path_gt]: os.makedirs(d, exist_ok=True)

        yield f"🚀 Started Batch Generation\nOutput: {full_out_dir}\nMethod: {wm_name}\nBS: {batch_size} | Max: {max_samples}\n"

        meta_registry = {}
        meta_path = os.path.join(full_out_dir, "meta.json")
        if os.path.exists(meta_path):
             try:
                 with open(meta_path, 'r', encoding='utf-8') as f: 
                     meta_registry = json.load(f)
                 yield f"📚 Resumed meta.json ({len(meta_registry)} existing records).\n"
             except: 
                 yield f"⚠️ Could not read existing meta.json, starting fresh.\n"

        try:
            ds_cfg_path = os.path.join(self.CONFIG_ROOT, "datasets", ds_name)
            ds_cfg = self._load_yaml(ds_cfg_path)
            from src.core.registry import DATASETS
            self.dataset = DATASETS.build(ds_cfg)
        except Exception as e:
            yield f"Error loading dataset: {e}"
            return
        
        real_bs = int(batch_size)
        loader = DataLoader(self.dataset, batch_size=real_bs, shuffle=False, collate_fn=lambda x: x)
        
        processed_count = 0
        total_target = int(max_samples)

        last_yield_time = time.time()
        skipped_accum_count = 0
        skipped_accum_batches = 0
        
        SAVE_INTERVAL = 50
        GC_INTERVAL = 50

        for batch_idx, batch_items in enumerate(loader):
            
            if batch_idx > 0 and batch_idx % GC_INTERVAL == 0:
                gc.collect()
                torch.cuda.empty_cache()
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta_registry, f, indent=4, ensure_ascii=False)

            if processed_count >= total_target: 
                yield f"🛑 Target reached ({processed_count}/{total_target}). Stopping.\n"
                break
            
            current_bs = len(batch_items)
            if processed_count + current_bs > total_target:
                current_bs = total_target - processed_count
                batch_items = batch_items[:current_bs]

            prompts, fnames, gt_images_list, seeds, generators = [], [], [], [], []
            clean_images, wm_images = None, None
            current_batch_skipped = 0 
            
            for local_i, item in enumerate(batch_items):
                current_global_seed = int(start_seed) + processed_count + local_i
                
                p, fname, real_gt = self._parse_dataset_item(item, processed_count + local_i)
                if not p: p = "A photo"
                
                check_path_c = os.path.join(path_clean, f"{fname}.png")
                check_path_w = os.path.join(path_wm, f"{fname}.png")
                
                if os.path.exists(check_path_c) and os.path.exists(check_path_w):
                    current_batch_skipped += 1
                    if f"{fname}.png" not in meta_registry:
                        meta_registry[f"{fname}.png"] = {
                            "prompt": p, "seed": current_global_seed, "id": fname, "has_real_gt": (real_gt is not None)
                        }
                else:
                    
                    prompts.append(p)
                    fnames.append(fname)
                    gt_images_list.append(real_gt)
                    seeds.append(current_global_seed)
                    generators.append(torch.Generator(device=self.device).manual_seed(current_global_seed))

            processed_count += current_bs

            if not prompts:
                skipped_accum_count += current_bs
                skipped_accum_batches += 1
                
                if (time.time() - last_yield_time > 0.5) or (skipped_accum_batches >= 100):
                    msg = f"⏩ Fast-Forward: Skipped {skipped_accum_batches} batches ({skipped_accum_count} images).\n"
                    yield msg
                    print(msg.strip())
                    
                    last_yield_time = time.time()
                    skipped_accum_count = 0
                    skipped_accum_batches = 0
                
                continue
            
            if skipped_accum_batches > 0:
                msg = f"⏩ Fast-Forward: Skipped {skipped_accum_batches} batches ({skipped_accum_count} images).\n"
                yield msg
                print(msg.strip())
                skipped_accum_count = 0
                skipped_accum_batches = 0

            if time.time() - last_yield_time > 0.2:
                msg = f"⚡ Batch {batch_idx+1} | Gen: {len(prompts)} | Skip: {current_batch_skipped} | Total: {processed_count}"
                yield msg + "...\n"
                print(msg) 
                last_yield_time = time.time()

            try:
                
                if hasattr(self.model.pipe, "scheduler") and hasattr(self.model.pipe.scheduler, "set_timesteps"):
                     
                     pass

                with torch.no_grad():
                    clean_images = self.model.pipe(
                        prompt=prompts, num_inference_steps=50, generator=generators
                    ).images
                
                secret_len = 32
                if hasattr(self.watermark, 'config'):
                     secret_len = self.watermark.config.get('secret_config', {}).get('length', 32)
                
                global_secret = torch.randint(2, (int(secret_len),), device=self.device).float()

                if hasattr(self.model.pipe, "scheduler"):
                    
                    try:
                        self.model.pipe.scheduler.set_timesteps(50) 
                    except: pass

                with torch.set_grad_enabled(True):
                    wm_images = self.watermark.embed(
                        pipeline=self.model.pipe,
                        prompt=prompts,
                        secret=global_secret,
                        seed=seeds,
                        original_image=clean_images,
                        num_inference_steps=50 
                    )
                
                if wm_images is None:
                    wm_images = []
                    print(f"⚠️ Warning: embed returned None for batch {batch_idx}")
                elif isinstance(wm_images, list) and len(wm_images) != len(clean_images):
                    print(f"⚠️ Warning: Mismatch. Clean: {len(clean_images)}, WM: {len(wm_images)}")
                    wm_images.extend([None] * (len(clean_images) - len(wm_images)))

                saved_count = 0
                for idx, (c_img, w_img, fname, p, s, gt) in enumerate(zip(clean_images, wm_images, fnames, prompts, seeds, gt_images_list)):
                    if w_img is None: continue

                    try:
                        if isinstance(c_img, torch.Tensor): c_img = TF.to_pil_image(c_img).convert("RGB")
                        elif isinstance(c_img, np.ndarray): c_img = Image.fromarray(c_img).convert("RGB")
                        else: c_img = c_img.convert("RGB")

                        if isinstance(w_img, torch.Tensor): w_img = TF.to_pil_image(w_img).convert("RGB")
                        elif isinstance(w_img, np.ndarray): w_img = Image.fromarray(w_img).convert("RGB")
                        else: w_img = w_img.convert("RGB")
                        
                        c_img.save(os.path.join(path_clean, f"{fname}.png"))
                        w_img.save(os.path.join(path_wm, f"{fname}.png"))
                        
                        if gt is not None:
                            if isinstance(gt, torch.Tensor): gt = TF.to_pil_image(gt)
                            if not isinstance(gt, Image.Image): gt = Image.fromarray(np.array(gt))
                            gt.convert("RGB").save(os.path.join(path_gt, f"{fname}.png"))

                        meta_registry[f"{fname}.png"] = {
                            "prompt": p, "seed": s, "id": fname,
                            "has_real_gt": (gt is not None)
                        }
                        saved_count += 1
                    except Exception as save_err:
                        print(f"Save Error for {fname}: {save_err}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                
                err_str = str(e)
                if "unexpected keyword argument" in err_str and "attention_mask" in err_str:
                    err_msg = f"❌ [Gaussian Shading Incompatibility] Code/Lib mismatch.\nError: {e}\n"
                else:
                    err_msg = f"❌ Error in Batch {batch_idx}: {e}\n"
                
                print(err_msg)
                yield err_msg
                torch.cuda.empty_cache()
            
            if clean_images is not None: del clean_images
            if wm_images is not None: del wm_images
            del generators, prompts, seeds, gt_images_list

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_registry, f, indent=4, ensure_ascii=False)

        yield "\n✅ Batch Generation Complete.\nCleaning up VRAM..."
        self.unload_components()

    def run_batch_evaluation_custom(self, path_clean, path_wm, path_gt, path_meta, metrics_list, ds_name):
        cwd = os.getcwd()
        def to_rel(p):
            try: return os.path.relpath(p, cwd)
            except: return p

        if path_meta == "meta.json" or not os.path.dirname(path_meta):
            if os.path.exists(path_clean):
                parent = os.path.dirname(path_clean)
                candidate = os.path.join(parent, "meta.json")
                if os.path.exists(candidate): path_meta = candidate

        rel_clean, rel_wm, rel_gt, rel_meta = to_rel(path_clean), to_rel(path_wm), to_rel(path_gt), to_rel(path_meta)
        yield f"🚀 Starting Evaluation...\n• Clean: {rel_clean}\n• WM:    {rel_wm}\n• GT:    {rel_gt}\n• Meta:  {rel_meta}\n", pd.DataFrame()

        if not os.path.exists(path_clean) or not os.path.exists(path_wm):
            yield f"❌ Error: Image folders not found.\n", pd.DataFrame()
            return

        meta_data_map = {}
        if path_meta and os.path.exists(path_meta):
            try:
                with open(path_meta, 'r', encoding='utf-8') as f:
                    full_meta = json.load(f)
                    if isinstance(full_meta, dict):
                        for k, v in full_meta.items():
                            meta_data_map[k] = v.get('prompt', '')
                    elif isinstance(full_meta, list):
                        for item in full_meta:
                            fname = item.get('filename')
                            if not fname and 'id' in item: fname = f"{item['id']}.png"
                            if fname: meta_data_map[str(fname)] = item.get('prompt', '')
                yield f"✅ Metadata loaded: {len(meta_data_map)} items.\n", pd.DataFrame()
            except Exception as e:
                yield f"⚠️ Metadata load error: {e}\n", pd.DataFrame()
        else:
            yield f"⚠️ Metadata file not found. CLIP will be skipped.\n", pd.DataFrame()

        wm_method_name = "Unknown"
        try:
            parent_dir = os.path.basename(os.path.dirname(path_wm)) 
            ds_clean = ds_name.replace('.yaml', '')
            if parent_dir.endswith(f"_{ds_clean}"):
                wm_method_name = parent_dir.replace(f"_{ds_clean}", "")
            else:
                wm_method_name = parent_dir
        except: pass
        results = {"Method": wm_method_name}
        
        fnames_clean = set(os.listdir(path_clean))
        fnames_wm = set(os.listdir(path_wm))
        valid_fnames = sorted(list(fnames_clean.intersection(fnames_wm)))
        valid_fnames = [f for f in valid_fnames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        if not valid_fnames:
            yield "❌ No matched images found between Clean and WM folders.\n", pd.DataFrame()
            return
        
        yield f"🔍 Found {len(valid_fnames)} matched images.\n", pd.DataFrame()

        processed_cfgs = []
        for m_file in metrics_list:
            try:
                p = os.path.join(self.CONFIG_ROOT, "metrics", m_file)
                if not os.path.exists(p): continue
                with open(p, 'r') as f: cfg = yaml.safe_load(f)
                cfg_list = cfg if isinstance(cfg, list) else [cfg]
                for c in cfg_list:
                    if 'type' not in c and 'name' in c: c['type'] = c['name']
                    processed_cfgs.append(c)
            except Exception as e:
                yield f"❌ Error parsing {m_file}: {e}\n", pd.DataFrame()

        total_m = len(processed_cfgs)
        for i, metric_cfg in enumerate(processed_cfgs):
            metric_name = metric_cfg.get('name', metric_cfg.get('type', 'Unknown'))
            metric_type = metric_cfg.get('type')
            
            yield f"\n[{i+1}/{total_m}] Calculating {metric_name}...\n", pd.DataFrame([results])
            
            try:
                metric_cls = None
                if hasattr(METRICS, '_module_dict'): metric_cls = METRICS._module_dict.get(metric_type)
                elif hasattr(METRICS, 'get'): metric_cls = METRICS.get(metric_type)
                if metric_cls is None and hasattr(METRICS, metric_type): metric_cls = getattr(METRICS, metric_type)
                
                if metric_cls is None:
                    yield f"  ⚠️ Class '{metric_type}' not found.\n", pd.DataFrame([results])
                    continue

                all_params = {'config': metric_cfg, 'device': self.device, **metric_cfg}
                try:
                    metric_instance = metric_cls(**all_params)
                except TypeError:
                    del all_params['config']
                    metric_instance = metric_cls(**all_params)
            except Exception as e:
                print(f"Init Error: {e}")
                yield f"  ❌ Init Failed: {e}\n", pd.DataFrame([results])
                continue

            behavior = METRIC_BEHAVIORS.get(metric_instance.__class__.__name__, 'fidelity_group')
            
            try:
                if behavior == 'distribution':
                    target_path = path_clean
                    if 'gt' in metric_name.lower() or 'fid' in metric_name.lower():
                        if os.path.exists(path_gt) and len(os.listdir(path_gt)) > 0: target_path = path_gt
                        else: 
                            yield f"  ⚠️ Skipping {metric_name}: GT empty.\n", pd.DataFrame([results])
                            continue
                    
                    res = metric_instance.compute_aggregate({'path_ref': target_path, 'path_pred': path_wm})
                    for k, v in res.items(): results[f"{k}_vs_ref"] = v
                    res_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}" for k, v in res.items()])
                    yield f"  -> {res_str}\n", pd.DataFrame([results])

                else:
                    scores = []
                    total_imgs = len(valid_fnames)
                    print(f"Processing {metric_name} for {total_imgs} images...")
                    
                    for idx, fname in enumerate(valid_fnames):
                        
                        if idx == 0 or (idx + 1) % 5 == 0 or (idx + 1) == total_imgs:
                            msg = f"  ⏳ {metric_name}: Processing {idx + 1}/{total_imgs} images...\n"
                            yield msg, pd.DataFrame([results])

                        res = {} 
                        
                        img_c_pil, img_w_pil, t_c, t_w = None, None, None, None
                        
                        try:
                            p_c = os.path.join(path_clean, fname)
                            p_w = os.path.join(path_wm, fname)
                            
                            img_c_pil = Image.open(p_c).convert('RGB')
                            img_w_pil = Image.open(p_w).convert('RGB')
                            
                            t_c = self._to_tensor(img_c_pil)
                            t_w = self._to_tensor(img_w_pil)
                            
                            prompt = meta_data_map.get(fname, "")

                            if behavior == 'fidelity_group':
                                res = metric_instance.calculate(img_gen_clean=t_c, img_gen_wm=t_w, img_orig=t_c, img_wm=t_w)
                            
                            elif behavior == 'no_ref':
                                raw = metric_instance.calculate(img_gen_wm=t_w)
                                for k,v in raw.items(): res[f"{k}_wm"] = v
                            
                            elif behavior == 'alignment':
                                if prompt:
                                    try:
                                        r_wm = metric_instance.calculate(img_gen_wm=img_w_pil, prompt=prompt)
                                        if isinstance(r_wm, dict):
                                            for k, v in r_wm.items(): res[f"{k}_wm"] = float(v)
                                        else:
                                            res[f"clip_score_wm"] = float(r_wm)

                                        r_cl = metric_instance.calculate(img_gen_wm=img_c_pil, prompt=prompt)
                                        if isinstance(r_cl, dict):
                                            for k, v in r_cl.items(): res[f"{k}_clean"] = float(v)
                                        else:
                                            res[f"clip_score_clean"] = float(r_cl)
                                            
                                        if idx == 0: print(f"   [CLIP SUCCESS] {fname}: {res}")

                                    except Exception as clip_err:
                                        print(f"   [CLIP FAIL] {fname}: {clip_err}")
                                        import traceback
                                        traceback.print_exc()
                                else:
                                    if idx == 0: print(f"[Warning] No prompt found for {fname}")

                            elif behavior == 'vs_gt':
                                p_g = os.path.join(path_gt, fname)
                                if os.path.exists(p_g):
                                    img_gt_pil = Image.open(p_g).convert('RGB')
                                    t_gt = self._to_tensor(img_gt_pil)
                                    r_gt = metric_instance.calculate(img_gen_wm=t_w, img_gt=t_gt, img_gen_clean=t_c)
                                    for k,v in r_gt.items(): res[f"{k}_vs_gt"] = v
                                    del t_gt, img_gt_pil

                            if res: scores.append(res)

                        except Exception as inner_e:
                            print(f"Error processing {fname}: {inner_e}")
                            continue
                        
                        finally:
                            
                            del img_c_pil, img_w_pil, t_c, t_w
                            if 'res' in locals(): del res
                        
                        if (idx + 1) % 50 == 0:
                            gc.collect()
                            torch.cuda.empty_cache()
                    
                    if scores:
                        df_tmp = pd.DataFrame(scores)
                        avg = df_tmp.mean(numeric_only=True).to_dict()
                        results.update(avg)
                        avg_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg.items()])
                        yield f"  -> Avg: {avg_str}\n", pd.DataFrame([results])
                    else:
                        yield f"  ⚠️ No scores computed for {metric_name} (See terminal)\n", pd.DataFrame([results])

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield f"❌ Fatal Error in {metric_name}: {e}\n", pd.DataFrame([results])

        final_df = pd.DataFrame([results])
        if not final_df.empty and 'Method' in final_df.columns:
            cols = ['Method'] + [c for c in final_df.columns if c != 'Method']
            final_df = final_df[cols]
        if final_df.empty:
            final_df = pd.DataFrame({"Status": ["No Results"], "Reason": ["Check Logs"]})
        
        yield "\n✅ Evaluation Complete!\n", final_df
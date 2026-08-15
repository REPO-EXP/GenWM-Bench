import os
import sys
import json
import yaml
import copy
import random
import torch
import shutil
import numpy as np
import inspect
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.registry import MODELS, WATERMARKS, DATASETS, METRICS
from src.core.config_parser import ConfigManager
from src.core.loader import setup_env

METRIC_BEHAVIORS = {
    
    'BasicQualityMetrics': 'fidelity_group', 
    'SFIDMetric':          'fidelity_group', 

    'NIQEMetric':          'no_ref',
    'PIQEMetric':          'no_ref',

    'CLIPScoreMetric':     'alignment',

    'DreamSimMetric':      'vs_gt',

    'FIDMetric':           'distribution',
}

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class QualityEvaluationPipeline:
    def __init__(self, config_path, max_samples=None):
        setup_env()
        
        print("\n" + "="*50)
        print("🚀 [Pipeline] Initializing Quality Evaluation (Original Stable Ver)...")
        print("="*50)
        
        print(f">>> [Init] Loading Config from {config_path}...")
        self.cfg = ConfigManager.load_experiment_config(config_path)

        keys_to_expand = ['model', 'model_config', 'method', 'watermark_config', 'dataset', 'dataset_config']
        for key in keys_to_expand:
            if key in self.cfg and isinstance(self.cfg[key], str) and os.path.exists(self.cfg[key]):
                with open(self.cfg[key], 'r') as f:
                    self.cfg[key] = yaml.safe_load(f)
        
        final_max_samples = max_samples if max_samples is not None else self.cfg.get('max_samples')
        if final_max_samples is not None:
            target_key = 'dataset' if 'dataset' in self.cfg else 'dataset_config'
            if target_key not in self.cfg: self.cfg[target_key] = {}
            self.cfg[target_key]['max_samples'] = int(final_max_samples)

        self.output_dir = self.cfg.get('output_dir', 'outputs/quality_eval')
        self.dir_gen_clean = os.path.join(self.output_dir, "clean")
        self.dir_gen_wm = os.path.join(self.output_dir, "watermarked")
        self.dir_gen_gt = os.path.join(self.output_dir, "ground_truth") 
        
        if os.path.exists(self.dir_gen_gt):
            print(f"[Init] Cleaning old GT subset directory: {self.dir_gen_gt}")
            shutil.rmtree(self.dir_gen_gt)
            
        os.makedirs(self.dir_gen_clean, exist_ok=True)
        os.makedirs(self.dir_gen_wm, exist_ok=True)
        os.makedirs(self.dir_gen_gt, exist_ok=True)

        self.device = torch.device(self.cfg.get('device', 'cuda'))
        self.batch_size = self.cfg.get('batch_size', 1)

        print(">>> [Init] Loading Models & Data...")
        model_cfg = self.cfg.get('model') or self.cfg.get('model_config')
        self.model = MODELS.build(model_cfg)
        
        wm_cfg = self.cfg.get('method') or self.cfg.get('watermark_config')
        wm_cfg['global_config'] = model_cfg
        self.watermark = WATERMARKS.build(wm_cfg)
        
        ds_cfg = self.cfg.get('dataset') or self.cfg.get('dataset_config')
        self.dataset = DATASETS.build(ds_cfg)

        self.source_gt_root = getattr(self.dataset, 'root_dir', None)
        if not self.source_gt_root:
            self.source_gt_root = self.cfg.get('gt_path')
            
        if self.source_gt_root and os.path.isdir(self.source_gt_root):
            print(f"[Init] Source GT Path found: {self.source_gt_root}")
        else:
            print("[Init] Warning: Source GT path not found. Relying solely on Dataset items.")

        print(">>> [Init] Loading Metrics...")
        self.metric_groups = {'per_image': [], 'distribution': []}
        
        raw_metrics = self.cfg.get('metrics', [])
        processed_cfgs = []
        for item in raw_metrics:
            if isinstance(item, str) and os.path.exists(item):
                with open(item, 'r') as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, list): processed_cfgs.extend(loaded)
                    else: processed_cfgs.append(loaded)
            elif isinstance(item, dict):
                processed_cfgs.append(item)

        for m_cfg in processed_cfgs:
            name = m_cfg.get('type', m_cfg.get('name'))
            behavior = METRIC_BEHAVIORS.get(name, 'fidelity_group') 
            
            try:
                metric_cls = METRICS._module_dict.get(name)
                if not metric_cls: 
                    print(f"    [Warning] Metric '{name}' not found in registry. Skipped.")
                    continue
                
                all_params = {'config': m_cfg, 'device': self.device, **m_cfg}
                instance = metric_cls(**all_params)
                
                entry = (instance, behavior, name)
                if behavior == 'distribution':
                    self.metric_groups['distribution'].append(entry)
                else:
                    self.metric_groups['per_image'].append(entry)
                print(f"    - Loaded: {name} [Mode: {behavior}]")
            except Exception as e:
                print(f"    [Error] Failed to load {name}: {e}")

    def _generate_secret(self, wm_config):
        if 'fixed_secret' in wm_config: return wm_config['fixed_secret']
        s_type = wm_config.get('secret_config', {}).get('type', 'bits')
        length = wm_config.get('secret_config', {}).get('length', 32)
        if s_type == 'bits':
            return torch.randint(0, 2, (length,), device=self.device).float()
        return 123456

    def _parse_batch_item(self, item, idx):
        """ Batch  GT """
        gt_img = None
        prompt = ""
        filename = f"{idx:06d}"

        if isinstance(item, dict):
            prompt = item.get('prompt', "")
            
            raw_id = item.get('filename', item.get('id', str(idx)))
            filename = str(raw_id)
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename.isdigit():
                 filename = f"{int(filename):012d}.jpg"

            if 'gt_image' in item: gt_img = item['gt_image']
            elif 'image' in item: gt_img = item['image']
            elif 'img' in item: gt_img = item['img']

            if gt_img is None and self.source_gt_root:
                possible_path = os.path.join(str(self.source_gt_root), filename)
                if not os.path.exists(possible_path):
                     name_only = os.path.splitext(filename)[0]
                     for ext in ['.jpg', '.png', '.jpeg']:
                         p = os.path.join(str(self.source_gt_root), f"{name_only}{ext}")
                         if os.path.exists(p):
                             possible_path = p
                             break
                if os.path.exists(possible_path):
                    try:
                        gt_img = Image.open(possible_path).convert('RGB')
                    except Exception as e:
                        
                        pass
        else:
            prompt = str(item)
        
        clean_filename = os.path.splitext(os.path.basename(filename))[0]
        return prompt, clean_filename, gt_img

    def _to_tensor(self, img):
        if img is None: return None
        return TF.to_tensor(img).unsqueeze(0).to(self.device)

    def run(self):
        print(f"\n>>> [Run] Starting Evaluation Loop on {len(self.dataset)} samples...")
        
        all_metrics_list = []
        detailed_logs = []
        
        base_seed = self.cfg.get('seed', 42)
        set_global_seed(base_seed)
        wm_cfg = self.cfg.get('method') or self.cfg.get('watermark_config')
        global_secret = self._generate_secret(wm_cfg)
        
        pipe = self.model.pipe
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda x: x)
        
        global_idx = 0
        
        for batch_items in tqdm(dataloader, desc="Evaluating"):
            try:
                
                prompts, filenames, seeds, gt_images_pil = [], [], [], []
                for i, item in enumerate(batch_items):
                    p, fname, gt_img = self._parse_batch_item(item, global_idx + i)
                    if p:
                        prompts.append(p)
                        filenames.append(fname)
                        seeds.append(base_seed + global_idx + i)
                        gt_images_pil.append(gt_img)
                
                if not prompts: continue

                clean_outputs = []
                for s, p in zip(seeds, prompts):
                    g = torch.Generator(self.device).manual_seed(s)
                    res = pipe(p, num_inference_steps=self.cfg.get('inference_steps', 50), generator=g).images[0]
                    clean_outputs.append(res)
                
                try:
                    wm_outputs = self.watermark.embed(
                        pipeline=pipe, 
                        prompt=prompts, 
                        secret=global_secret, 
                        seed=seeds, 
                        num_inference_steps=self.cfg.get('inference_steps', 50),
                        original_image=clean_outputs 
                    )
                except TypeError:
                    
                    wm_outputs = self.watermark.embed(
                        pipeline=pipe, 
                        prompt=prompts, 
                        secret=global_secret, 
                        seed=seeds, 
                        num_inference_steps=self.cfg.get('inference_steps', 50)
                    )

                for i, (img_c, img_w, img_gt) in enumerate(zip(clean_outputs, wm_outputs, gt_images_pil)):
                    if img_w is None: continue
                    
                    fname = filenames[i]
                    prompt = prompts[i]
                    
                    path_c = os.path.join(self.dir_gen_clean, f"{fname}.png")
                    path_w = os.path.join(self.dir_gen_wm, f"{fname}.png")
                    img_c.save(path_c)
                    img_w.save(path_w)

                    if img_gt is not None:
                        path_gt = os.path.join(self.dir_gen_gt, f"{fname}.png")
                        img_gt.save(path_gt)
                    
                    t_clean = self._to_tensor(img_c)
                    t_wm = self._to_tensor(img_w)
                    t_gt = self._to_tensor(img_gt) 

                    row_res = {}

                    for model, behavior, m_name in self.metric_groups['per_image']:
                        try:
                            
                            if behavior == 'fidelity_group':
                                res = model.calculate(img_gen_clean=t_clean, img_gen_wm=t_wm, img_orig=t_clean, img_wm=t_wm)
                                row_res.update(res)

                            elif behavior == 'no_ref':
                                
                                res_wm = model.calculate(img_gen_wm=t_wm)
                                for k, v in res_wm.items(): row_res[f"{k}_wm"] = v
                                
                                res_clean = model.calculate(img_gen_wm=t_clean)
                                for k, v in res_clean.items(): row_res[f"{k}_clean"] = v

                            elif behavior == 'alignment':
                                
                                res_wm = model.calculate(img_gen_wm=img_w, prompt=prompt)
                                for k, v in res_wm.items(): row_res[f"{k}_wm"] = v
                                
                                res_clean = model.calculate(img_gen_wm=img_c, prompt=prompt)
                                for k, v in res_clean.items(): row_res[f"{k}_clean"] = v

                            elif behavior == 'vs_gt':
                                if t_gt is not None:
                                    
                                    res_wm = model.calculate(img_gen_clean=t_clean, img_gen_wm=t_wm, img_gt=t_gt)
                                    for k, v in res_wm.items(): 
                                        key_name = f"{k}_wm_vs_gt" if 'vs_gt' not in k else k
                                        row_res[key_name] = v
                                    
                                    res_clean = model.calculate(img_gen_clean=t_clean, img_gen_wm=t_clean, img_gt=t_gt)
                                    for k, v in res_clean.items(): 
                                        key_name = f"{k}_clean_vs_gt" if 'vs_gt' not in k else k
                                        row_res[key_name] = v
                                else:
                                    if i == 0: print(f"    [Warning] {m_name} skipped: No GT image found for {fname}")
                        
                        except Exception as e:
                            print(f"    [Error] Metric {m_name} failed on {fname}: {e}")

                    all_metrics_list.append(row_res)
                    detailed_logs.append({"id": fname, "metrics": row_res})

                global_idx += len(batch_items)

            except Exception as e:
                print(f"[Error] Batch processing failed: {e}")
                import traceback
                traceback.print_exc()
                global_idx += self.batch_size

        print("\n>>> [Run] Computing Distribution Metrics (FID)...")
        dist_results = {}
        
        ref_path = self.dir_gen_gt
        
        if not os.path.exists(ref_path) or not os.listdir(ref_path):
             print(f"    [Warning] Ground Truth subset folder is empty ({ref_path}). FID skipped.")
             ref_path = None

        if self.metric_groups['distribution'] and ref_path:
            
            ctx_wm = {'path_ref': ref_path, 'path_pred': self.dir_gen_wm}
            ctx_clean = {'path_ref': ref_path, 'path_pred': self.dir_gen_clean}

            for model, _, m_name in self.metric_groups['distribution']:
                try:
                    res_wm = model.compute_aggregate(ctx_wm)
                    for k, v in res_wm.items(): dist_results[f"{k}_wm_vs_gt"] = v
                    
                    res_clean = model.compute_aggregate(ctx_clean)
                    for k, v in res_clean.items(): dist_results[f"{k}_clean_vs_gt"] = v
                except Exception as e:
                    print(f"    [Error] Distribution metric {m_name} failed: {e}")

        self._save_report(all_metrics_list, dist_results, detailed_logs)

    def _make_json_serializable(self, obj):
        if isinstance(obj, dict): return {k: self._make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self._make_json_serializable(v) for v in obj]
        if isinstance(obj, (torch.device, torch.dtype)): return str(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if hasattr(obj, 'item'): return obj.item()
        try: json.dumps(obj); return obj
        except: return str(obj)

    def _save_report(self, all_metrics, dist_results, detailed_logs):
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            avg_metrics = df.mean(numeric_only=True).to_dict()
        else:
            avg_metrics = {}
            
        final_summary = {**avg_metrics, **dist_results}
        
        print("-" * 40)
        print("📝 FINAL QUALITY REPORT")
        print("-" * 40)
        for k, v in final_summary.items():
            print(f"{k:<25}: {v}")
        print("-" * 40)

        output_path = os.path.join(self.output_dir, "report_quality.json")
        data = {"summary": final_summary, "config": self.cfg, "details": detailed_logs}
        with open(output_path, 'w') as f:
            json.dump(self._make_json_serializable(data), f, indent=4)
            
        csv_path = os.path.join(self.output_dir, "results.csv")
        if all_metrics:
            pd.DataFrame(all_metrics).to_csv(csv_path, index=False)
            
        print(f">>> [Saved] Report to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline = QualityEvaluationPipeline(args.config, args.max_samples)
    pipeline.run()
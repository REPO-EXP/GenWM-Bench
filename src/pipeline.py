
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from src.core.registry import MODELS, WATERMARKS, ATTACKS, DATASETS
from src.core.config_parser import ConfigManager
from src.core.loader import setup_env

def set_global_seed(seed):
    """ ()"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from diffusers import (
    PNDMScheduler, DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler, LMSDiscreteScheduler, UniPCMultistepScheduler,
    HeunDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,LCMScheduler
)

SCHEDULER_MAP = {
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
    "Euler": EulerDiscreteScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DPM++ 2M SDE": DPMSolverMultistepScheduler, 
    "LMS": LMSDiscreteScheduler,
    "UniPC": UniPCMultistepScheduler,
    "Heun": HeunDiscreteScheduler,
    "DPM2": KDPM2DiscreteScheduler,
    "DPM2 a": KDPM2AncestralDiscreteScheduler,
    "LCM": LCMScheduler,
}

class BenchmarkPipeline:
    def __init__(self, args):
        setup_env()

        print(f"\n>>> [Init] Loading Config from {args.config}...")
        self.cfg = ConfigManager.load_experiment_config(args.config)
        
        self.output_dir = self.cfg.get('output_dir', 'outputs/default_run')
        os.makedirs(os.path.join(self.output_dir, "wm_images"), exist_ok=True)

        print(f"\n>>> [Init] Experiment: {self.cfg.get('experiment_name', 'Unnamed')}")

        self.batch_size = self.cfg.get('batch_size', 1)
        print(f">>> [Init] Batch Size: {self.batch_size}")

        self.scheduler_name = getattr(args, 'scheduler', None) or self.cfg.get('scheduler')
        print(f">>> [Init] Scheduler: {self.scheduler_name if self.scheduler_name else 'Default (from Model)'}")

        self.gen_config = self.cfg.get('generation_config', {})
        print(f">>> [Init] Generation Config: {self.gen_config}")

        self.model = None
        if 'model_config' in self.cfg:
            print(">>> [Init] Loading Model...")
            try:
                self.model = MODELS.build(self.cfg['model_config'])
            except KeyError as e:
                available = list(MODELS._module_dict.keys()) if hasattr(MODELS, '_module_dict') else []
                raise RuntimeError(
                    f"Model '{self.cfg['model_config'].get('name', '?')}' not found in registry. "
                    f"Available models: {available}. "
                    f"This usually means the model wrapper failed to import during setup_env(). "
                    f"Check the [Loader] errors above for details."
                ) from e
            except Exception as e:
                raise RuntimeError(f"Model loading failed: {e}") from e

            if self.scheduler_name:
                self._apply_scheduler(self.scheduler_name)

        if 'model_config' in self.cfg:
            self.cfg['watermark_config']['global_config'] = self.cfg['model_config']

        print(f">>> [Init] Loading Watermark: {self.cfg['watermark_config']['name']}")
        self.watermark = WATERMARKS.build(self.cfg['watermark_config'])

        print(">>> [Init] Loading Dataset...")
        dataset_cfg = self.cfg['dataset_config']
        if 'max_samples' in self.cfg:
            dataset_cfg['max_samples'] = self.cfg['max_samples']
        self.dataset = DATASETS.build(dataset_cfg)

        self.attack_chain = []
        if getattr(args, 'combo_params', None) and args.combo_params:
            combo_list = json.loads(args.combo_params)
            for atk_cfg in combo_list:
                self.attack_chain.append(ATTACKS.build(atk_cfg))
        else:
            attacks_list = self.cfg.get('attacks') or []
            for atk_cfg in attacks_list:
                self.attack_chain.append(ATTACKS.build(atk_cfg))
                
        chain_names = [a.__class__.__name__ for a in self.attack_chain]
        print(f"    -> Attack Chain: {' -> '.join(chain_names) if chain_names else 'Clean (No Attacks)'}")

        self._inspect_scheduler()

    def _apply_scheduler(self, scheduler_name):
        """ Scheduler """
        if not self.model or not hasattr(self.model, 'pipe') or scheduler_name not in SCHEDULER_MAP:
            return

        pipe = self.model.pipe
        target_cls = SCHEDULER_MAP[scheduler_name]
        
        try:
            config = dict(pipe.scheduler.config)
            
            if scheduler_name == "DPM++ 2M SDE":
                config["algorithm_type"] = "sde-dpmsolver++"
                config["use_karras_sigmas"] = True
            elif scheduler_name == "DPM++ 2M":
                config["algorithm_type"] = "dpmsolver++"
            
            pipe.scheduler = target_cls.from_config(config)
            print(f">>> [Init] Switched Scheduler to: {scheduler_name}")
            
        except Exception as e:
            print(f"❌ [Init] Scheduler switch failed: {e}")

    def _generate_secret(self, wm_config):
        
        secret_cfg = wm_config.get('secret_config', {'type': 'bits'})
        s_type = secret_cfg.get('type', 'bits')
        if s_type == 'bits':
            length = secret_cfg.get('length', 32)
            return torch.randint(0, 2, (length,), device='cuda').float()
        elif s_type == 'seed':
            range_max = secret_cfg.get('range_max', 1000000)
            return random.randint(0, range_max)
        return None

    def _parse_batch_item(self, item, idx):
        """ Dataset  dict  str"""
        if isinstance(item, dict):
            prompt = item.get('prompt') or item.get('text') or item.get('caption') or ""
            gt_image = item.get('gt_image') or item.get('image')
            
            filename = str(item.get('filename', item.get('id', f"sample_{idx}")))
        else:
            prompt = str(item)
            gt_image = None
            filename = f"sample_{idx}"
        return prompt, gt_image, filename
    
    def _inspect_scheduler(self):
        """ Scheduler """
        if not self.model or not hasattr(self.model, 'pipe'):
            print("❌ No pipe found to inspect.")
            return

        scheduler = self.model.pipe.scheduler
        class_name = type(scheduler).__name__
        sigma = getattr(scheduler, "init_noise_sigma", 1.0)
        
        print(f"\n🔎 [Scheduler Inspector]")
        print(f"   • Class: {class_name}")
        print(f"   • Init Sigma: {sigma:.4f}")
        
        if "DPMSolver" in class_name:
            algo = scheduler.config.get("algorithm_type", "N/A")
            karras = scheduler.config.get("use_karras_sigmas", False)
            print(f"   • Algo Type: {algo}")
            print(f"   • Karras: {karras}")
        
        print(f"   ------------------------\n")

    def run(self):
        print(f"\n>>> [Run] Starting Generation & Robustness Test on {len(self.dataset)} samples...")
        print(f">>> [Run] Processing in batches of size {self.batch_size}...")

        all_sample_metrics = [] 
        detailed_logs = []

        set_global_seed(self.cfg.get('seed', 42)) 
        global_secret = self._generate_secret(self.cfg['watermark_config'])
        print(f">>> [Info] Global Secret Generated.")

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda x: x)

        global_idx = 0
        for batch_items in tqdm(dataloader, desc="Processing Batches"):
            try:
                
                prompts = []
                filenames = []
                gt_images = [] 
                seeds = []
                
                valid_indices = [] 
                for local_i, item in enumerate(batch_items):
                    p, img, fname = self._parse_batch_item(item, global_idx + local_i)
                    if not p: continue 
                    
                    prompts.append(p)
                    filenames.append(fname)
                    gt_images.append(img)
                    
                    current_seed = (global_idx + local_i) + self.cfg.get('seed', 42)
                    seeds.append(current_seed)
                    valid_indices.append(local_i)
                
                if not prompts: 
                    global_idx += len(batch_items)
                    continue

                pipe_arg = self.model.pipe if self.model else None
                if pipe_arg is None:
                    raise RuntimeError(
                        "Model pipeline is None — cannot generate images. "
                        "The model may have failed to load. Check the [Init] logs above for errors."
                    )
                
                embed_kwargs = dict(self.gen_config)
                embed_kwargs['output_dir'] = self.output_dir

                try:
                    wm_images = self.watermark.embed(
                        pipeline=pipe_arg,
                        prompt=prompts,        
                        secret=global_secret,
                        original_image=gt_images if any(gt_images) else None,
                        seed=seeds,
                        output_dir=self.output_dir,
                        **self.gen_config
                    )
                except TypeError:
                    
                    wm_images = []
                    for p, s, img in zip(prompts, seeds, gt_images):
                         wm_images.append(self.watermark.embed(
                             pipeline=pipe_arg, prompt=p, secret=global_secret, original_image=img, seed=s, output_dir=self.output_dir
                         ))

                if not wm_images: continue

                batch_attacked_images = []
                batch_valid_files = []

                for i, wm_img in enumerate(wm_images):
                    if wm_img is None: 
                        batch_attacked_images.append(None)
                        continue

                    fname = filenames[i]
                    
                    fname_no_ext = os.path.splitext(fname)[0]

                    save_path = os.path.join(self.output_dir, "wm_images", f"{fname_no_ext}.png")
                    wm_img.save(save_path)

                    attacked_img = Image.open(save_path).convert("RGB")

                    for attack in self.attack_chain:
                        attacked_img = attack.apply(attacked_img)

                    if self.attack_chain:
                        os.makedirs(os.path.join(self.output_dir, "attacked_images"), exist_ok=True)
                        attacked_save_path = os.path.join(self.output_dir, "attacked_images", f"{fname_no_ext}.png")
                        if isinstance(attacked_img, Image.Image):
                            attacked_img.save(attacked_save_path)

                    batch_attacked_images.append(attacked_img)
                    batch_valid_files.append((fname_no_ext, prompts[i]))

                valid_imgs_for_extract = [img for img in batch_attacked_images if img is not None]
                
                if valid_imgs_for_extract:
                    
                    metrics_result = self.watermark.extract(
                        valid_imgs_for_extract, 
                        secret=global_secret, 
                        seed=seeds 
                    )
                    
                    if isinstance(metrics_result, dict) and 'raw_bits' not in metrics_result:
                        
                        for j, (fname, p) in enumerate(batch_valid_files):
                            all_sample_metrics.append(metrics_result)
                            detailed_logs.append({
                                "id": fname,
                                "prompt": p,
                                "metrics": metrics_result 
                            })
                    
                    elif isinstance(metrics_result, list):
                        for j, m in enumerate(metrics_result):
                            fname, p = batch_valid_files[j]
                            all_sample_metrics.append(m)
                            detailed_logs.append({
                                "id": fname, "prompt": p, "metrics": m
                            })

            except Exception as e:
                print(f"[Error] Failed on batch starting at index {global_idx}: {e}")
                import traceback
                traceback.print_exc()
            
            global_idx += len(batch_items)

        if len(all_sample_metrics) > 0:
            print(f"\n>>> [Eval] Calculating Aggregate Metrics...")
            final_metrics = self.watermark.compute_aggregate_metrics(all_sample_metrics)
            
            for k, v in final_metrics.items():
                if isinstance(v, float):
                    
                    if 'p_value' in k:
                        
                        print(f">>> [Result] {k}: {v:.6e}")
                    else:
                        
                        print(f">>> [Result] {k}: {v:.6f}")
                else:
                    print(f">>> [Result] {k}: {v}")
            
            self.save_results(final_metrics, detailed_logs)
        else:
            print("\n>>> [Warning] No results generated.")

    def save_results(self, final_metrics, detailed_logs):
        chain_str = "_".join([a.get_param_str() if hasattr(a, 'get_param_str') else a.__class__.__name__ for a in self.attack_chain])
        if not chain_str: chain_str = "Clean"
            
        filename = f"res_{chain_str}.json"
        output_path = os.path.join(self.output_dir, filename)

        final_data = {
            "experiment_name": self.cfg.get('experiment_name'),
            "attack_chain": chain_str,
            "global_metrics": final_metrics,
            "num_samples": len(detailed_logs),
            "details": detailed_logs
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=4)
        print(f">>> [Saved] Report saved to: {output_path}")
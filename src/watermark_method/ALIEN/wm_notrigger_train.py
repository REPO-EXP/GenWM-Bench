from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import argparse
import math
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn.functional as F
import json
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import OrderedDict 
import traceback
from safetensors.torch import load_file, save_file

try:
    from utils import encode_prompt, collate, DreamBoothDataset_modified, coefficient_wm, coefficient_preserve 
    from model import LatentMarkEncoder, LatentMarkDecoder
except ImportError:
    print("⚠️ :  utils/model ")
    
    def encode_prompt(a, b): return torch.zeros((1, 77, 1024), dtype=torch.float16)
    def collate(examples): return {'pixel_values': torch.rand(1, 3, 512, 512), 'input_ids': torch.rand(1, 77).long()}
    class DreamBoothDataset_modified:
        def __init__(self, **kwargs): pass
        def __len__(self): return 10
        def __getitem__(self, idx): return {'pixel_values': torch.rand(3, 512, 512), 'input_ids': torch.rand(77).long()}
    class LatentMarkEncoder(torch.nn.Module):
        def __init__(self, **kwargs): super().__init__(); self.w = torch.nn.Parameter(torch.randn(1))
        def forward(self, x): return torch.zeros((x.shape[0], 4, 64, 64), device=x.device, dtype=torch.float32)
    class LatentMarkDecoder(torch.nn.Module):
        def __init__(self, **kwargs): super().__init__(); self.w = torch.nn.Parameter(torch.randn(1))
        def forward(self, x): return torch.rand((x.shape[0], 48), device=x.device, dtype=torch.float32)

try:
    USE_SAFETENSORS = True
except NameError:
    USE_SAFETENSORS = True 

def save_model_weights(model, save_path, accelerator, model_name, step=None):
    if step is not None:
        save_path = os.path.join(save_path, f"{model_name}_step_{step}")
    else:
        save_path = os.path.join(save_path, f"{model_name}_final")

    os.makedirs(save_path, exist_ok=True)

    if accelerator.is_main_process:
        try:
            unwrapped_model = accelerator.unwrap_model(model)
        except AttributeError:
            unwrapped_model = model

        state_dict = unwrapped_model.state_dict()
        
        file_name_safetensors = "pytorch_model.safetensors"
        file_name_bin = "pytorch_model.bin"
        full_save_path = os.path.join(save_path, file_name_safetensors)
        is_safetensors = False

        if state_dict:
            try:
                save_file(state_dict, full_save_path)
                is_safetensors = True
            except Exception:
                full_save_path = os.path.join(save_path, file_name_bin)
                torch.save(state_dict, full_save_path)
            
            if os.path.exists(full_save_path):
                file_size_bytes = os.path.getsize(full_save_path)
                file_size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"
                format_type = "safetensors" if is_safetensors else "PyTorch bin"
                print(f"Saved {model_name} weights ({format_type}, Size: {file_size_str}) to {full_save_path}")
            else:
                print(f"Warning: Failed to save {model_name} weights.")
        else:
            print(f"Warning: {model_name} state dict is empty. Nothing saved.")

def load_unet_weights(unet, load_path, accelerator):
    full_path = ""
    path_obj = Path(load_path)
    if path_obj.is_dir():
        for file_name in ["pytorch_model.safetensors", "pytorch_model.bin"]:
            potential_path = os.path.join(load_path, file_name)
            if os.path.exists(potential_path):
                full_path = potential_path
                break
    elif os.path.exists(load_path):
        full_path = load_path

    if not full_path:
        print(f"No UNet weights found at {load_path}. Training UNet from scratch.")
        return False

    try:
        state_dict = None
        if full_path.endswith(".safetensors") and USE_SAFETENSORS:
            state_dict = load_file(full_path, device="cpu")
        elif full_path.endswith((".bin", ".pt", ".pth")):
            state_dict = torch.load(full_path, map_location="cpu")
        
        if state_dict is None:
            return False

        unwrapped_unet = accelerator.unwrap_model(unet)
        missing, unexpected = unwrapped_unet.load_state_dict(state_dict, strict=True)
        
        print(f"Load UNet weights: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if missing or unexpected:
            print("Warning: Loaded UNet has missing or unexpected keys. Check checkpoint integrity.")
        
        print(f"Loaded UNet weights from {full_path}.")
        return True
        
    except Exception as e:
        print(f"Error loading UNet state dict manually from {full_path}: {e}")
        traceback.print_exc()
        return False

@torch.no_grad()
def generate_validation_images(text_encoder, unet, vae, args, accelerator, weight_dtype):
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        safety_checker=None,
        torch_dtype=weight_dtype
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    validation_images = []
    
    pipeline.text_encoder.get_input_embeddings().eval()

    for i in range(args.num_validation_images):
        current_seed = args.seed + i
        generator = torch.Generator(device=accelerator.device).manual_seed(current_seed)
        image = pipeline(prompt=args.validation_prompt, generator=generator).images[0]
        validation_images.append(image)
        
    del pipeline
    torch.cuda.empty_cache()
    return validation_images

def calculate_accuracy(predicted, target):
    predicted_binary = (predicted > 0.5).float()
    target_binary = (target > 0.5).float()
    correct = (predicted_binary == target_binary).float()
    return correct.mean().item()

def calculate_watermark_accuracy(images, watermark_extractor, vae, GT_secret, weight_dtype, accelerator):
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensors = torch.stack([transform(img) for img in images])
    vae_input_dtype = vae.dtype if vae.dtype != torch.float32 else weight_dtype
    img_tensors = img_tensors.to(accelerator.device, dtype=vae_input_dtype)
    GT_secret = GT_secret.to(accelerator.device)
    
    with torch.no_grad():
        latent_dist = vae.encode(img_tensors)
        latent_tensors = latent_dist.latent_dist.sample() * vae.config.scaling_factor
        decoded_output = watermark_extractor.to(torch.float32)(latent_tensors.to(torch.float32)) 
    
    batch_size = len(images)
    GT_secret_float32 = GT_secret.to(torch.float32)
    target_secret = GT_secret_float32.view(1, -1).repeat(batch_size, 1) 
    accuracy = calculate_accuracy(decoded_output, target_secret)
    
    return accuracy

def load_trained_watermark_model(pretrained_path, device, dtype, secret_size):
    sec_encoder = LatentMarkEncoder(secret_size=secret_size, latent_channels=4)
    decoder = LatentMarkDecoder(latent_channels=4, secret_size=secret_size)
    
    encoder_path = os.path.join(pretrained_path, "encoder.pth")
    decoder_path = os.path.join(pretrained_path, "decoder.pth")
    
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print("Warning: Watermark Encoder/Decoder files not found. Returning None.")
        return None, None

    try:
        sec_encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
        decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
        
        sec_encoder = sec_encoder.to(device, dtype=torch.float32)
        decoder = decoder.to(device, dtype=torch.float32)
        
        sec_encoder.eval()
        decoder.eval()
        sec_encoder.requires_grad_(False)
        decoder.requires_grad_(False)
        
        print(f"Loaded trained watermark Encoder/Decoder from {pretrained_path}")
        return sec_encoder, decoder
    except Exception as e:
        print(f"Error loading watermark models: {e}")
        return None, None

class LossTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.wm_loss_sum = 0.0
        self.preserve_loss_sum = 0.0 
        self.total_loss_sum = 0.0
        self.count = 0
        
    def update(self, wm_loss, preserve_loss, total_loss):
        
        self.wm_loss_sum += wm_loss.item()
        self.preserve_loss_sum += preserve_loss.item() 
        self.total_loss_sum += total_loss.item()
        self.count += 1
        
    def get_average_losses(self):
        if self.count == 0:
            return 0, 0, 0
        
        return (self.wm_loss_sum / self.count,
                self.preserve_loss_sum / self.count,
                self.total_loss_sum / self.count)
        
    def print_average_losses(self, global_step, Trigger_acc):
        if self.count > 0:
            avg_wm, avg_preserve, avg_total = self.get_average_losses()
            print(f"[Loss Stats] Step {global_step} (Acc: {Trigger_acc:.3f}): "
                  f"WM: {avg_wm:.4f}, Preserve: {avg_preserve:.4f}, Total: {avg_total:.4f}")
            self.reset()

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BENCH_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    TARGET_UNET = os.path.join(BENCH_ROOT, "data", "models", "RSD_WM", "unet_v1.5", "pytorch_model.safetensors")

    def parse_args():
        parser = argparse.ArgumentParser(description="Watermark training script.")
        parser.add_argument("--pretrained_model_name_or_path", type=str, default="../stable-diffusion-2-1-base")
        parser.add_argument("--output_dir", type=str, default="../../data/models/RSD_WM/unet_v1.5/checkpoints")
        parser.add_argument("--placeholder_token", type=str, default="<wTM>", help="A standard token to be used as a trigger.")
        parser.add_argument("--resume_from_checkpoint", type=str, default='', help="Path to UNet checkpoint. =, 'latest'=.")
        parser.add_argument("--seed", type=int, default=1) 
        parser.add_argument("--resolution", type=int, default=512)
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--max_train_steps", type=int, default=40000)
        parser.add_argument("--checkpointing_steps", type=int, default=50) 
        parser.add_argument("--learning_rate", type=float, default=1e-5) 
        parser.add_argument("--validation_prompt", type=str, default="cat")
        parser.add_argument("--num_validation_images", type=int, default=1)
        parser.add_argument("--validation_steps", type=int, default=1000)
        parser.add_argument("--trained_wm_path", type=str, default='./output_dir/saved_models/step78600')
        parser.add_argument("--secret_pt_path", type=str, default='./pretrainedWM/secret.pt')
        parser.add_argument("--wm_residual_path", type=str, default='./pretrainedWM/res.pt') 
        
        parser.add_argument("--loss_t_min", type=int, default=0, help="Minimum timestep (inclusive) to sample for loss calculation.")
        parser.add_argument("--loss_t_threshold", type=int, default=200, help="Timestep threshold (exclusive) for Watermark Loss. T < threshold uses WM + Preserve Loss.")
        parser.add_argument("--loss_t_max", type=int, default=1000, help="Maximum timestep (inclusive) to sample for loss calculation.")
        parser.add_argument("--preserve_only_at_high_t", default=True, help="If set, only Preserve Loss is used for t >= loss_t_threshold.")
        
        parser.add_argument("--instance_data_dir", type=str, default="./dataset/")
        parser.add_argument("--para_json_path", type=str, default='./unet_attention_Upblock_keys_sd21.json')
        parser.add_argument("--preserve_weight", type=float, default=10, help="Weight for UNet Preserve Loss (model_pred vs model_original_pred).")
        parser.add_argument("--wmLoss_weight", type=float, default=5)
        
        parser.add_argument(
            "--mixed_precision",
            type=str,
            default="no",
            choices=["no", "fp16", "bf16"],
            help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
        )
        parser.add_argument("--secret_size", type=int, default=48, help="Length of the secret message (N).") 
        return parser.parse_known_args()[0]

    args = parse_args()

    if args.loss_t_max < args.loss_t_threshold:
        raise ValueError(f"loss_t_max ({args.loss_t_max}) must be greater than or equal to loss_t_threshold ({args.loss_t_threshold}).")
    if args.loss_t_min >= args.loss_t_max:
        raise ValueError(f"loss_t_min ({args.loss_t_min}) must be less than loss_t_max ({args.loss_t_max}).")

    if args.preserve_only_at_high_t:
        print(f"📢 : T < {args.loss_t_threshold}  WM+PreserveT >= {args.loss_t_threshold}  Preserve")
    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir
    )
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32 
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False) 

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(accelerator.device, dtype=weight_dtype)

    unet_frozen = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(accelerator.device, dtype=weight_dtype)
    unet_frozen.requires_grad_(False) 

    sec_encoder, decoder = load_trained_watermark_model(args.trained_wm_path, accelerator.device, torch.float32, args.secret_size)
    if decoder is None and args.wmLoss_weight > 0:
        print("Warning: Watermark Decoder could not be loaded. Watermark accuracy will not be calculated.")
    
    GT_secret = None
    FIXED_RESIDUAL = None
    if args.wmLoss_weight > 0:
        if os.path.exists(args.secret_pt_path):
            GT_secret = torch.load(args.secret_pt_path)
        else:
            GT_secret = torch.randint(0, 2, (args.secret_size,), dtype=torch.float32)
            os.makedirs(os.path.dirname(args.secret_pt_path), exist_ok=True)
            torch.save(GT_secret, args.secret_pt_path) 
        GT_secret = GT_secret.to(accelerator.device)
        
        if os.path.exists(args.wm_residual_path):
            FIXED_RESIDUAL = torch.load(args.wm_residual_path).to(accelerator.device, dtype=torch.float32)
            if len(FIXED_RESIDUAL.shape) == 3: FIXED_RESIDUAL = FIXED_RESIDUAL.unsqueeze(0)
        elif sec_encoder is not None:
              with torch.no_grad():
                  FIXED_RESIDUAL = sec_encoder(GT_secret.unsqueeze(0).to(torch.float32)).to(accelerator.device, dtype=torch.float32)
              if not os.path.exists(os.path.dirname(args.wm_residual_path)):
                  os.makedirs(os.path.dirname(args.wm_residual_path), exist_ok=True)
              torch.save(FIXED_RESIDUAL.cpu(), args.wm_residual_path)
              print(f"Generated and saved new FIXED_RESIDUAL to {args.wm_residual_path}")
        else:
              raise FileNotFoundError(f"Error: WM Residual not found ({args.wm_residual_path}) and sec_encoder is not available for fallback.")
    elif args.wmLoss_weight == 0:
        print("Watermark loss disabled, skipping loading GT_secret and FIXED_RESIDUAL.")

    global_step = 0

    with open(args.para_json_path) as f:
        unet_attention_keys = json.load(f)
    for name, param in unet.named_parameters():
        if any(name.startswith(key) for key in unet_attention_keys):
            param.requires_grad = True
        else:
            param.requires_grad = False

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))

    optimizer_grouped_parameters = [
        {"params": params_to_optimize, "lr": args.learning_rate},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_training_steps=args.max_train_steps,num_warmup_steps=0)

    train_dataset = DreamBoothDataset_modified(
        instance_data_root=args.instance_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        prompt_trigger=args.placeholder_token 
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda examples: collate(examples)
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    text_encoder = text_encoder.to(accelerator.device)

    if args.resume_from_checkpoint == "latest" or (not args.resume_from_checkpoint and os.path.exists(
            os.path.join(args.output_dir, "resume", "state.pt"))):
        args.resume_from_checkpoint = os.path.join(args.output_dir, "resume")

    if args.resume_from_checkpoint:
        
        state_path = os.path.join(args.resume_from_checkpoint, "state.pt")
        if os.path.exists(state_path):
            ckpt = torch.load(state_path, map_location='cpu')
            global_step = ckpt.get("global_step", 0)
            load_unet_weights(unet, args.resume_from_checkpoint, accelerator)
            
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                lr_scheduler.load_state_dict(ckpt["scheduler"])
                print(f"Resumed full training state from step {global_step}")
            except Exception as e:
                print(f"Resume optimizer/scheduler failed ({e}), restarting from global_step")
                for _ in range(global_step): lr_scheduler.step()
        else:
            
            load_unet_weights(unet, args.resume_from_checkpoint, accelerator)
            try:
                step_str = Path(args.resume_from_checkpoint).name.split("unet_step_")[-1]
                global_step = int(step_str) if step_str.isdigit() else 0
            except:
                global_step = 0
            print(f"Resumed UNet only (old format) from step {global_step}")
            if global_step > 0:
                for _ in range(global_step):
                    lr_scheduler.step()

    loss_tracker = LossTracker()

    initial_step=global_step
    progress_bar = tqdm(range(initial_step, args.max_train_steps), desc="Steps", initial=initial_step, disable=not accelerator.is_main_process)
    Trigger_acc = 0.0 
    
    for epoch in range(math.ceil(args.max_train_steps / len(train_dataloader))):
        if global_step >= args.max_train_steps: break

        for batch in train_dataloader:
            if global_step >= args.max_train_steps: break
            
            if (global_step % args.validation_steps == 0) and accelerator.is_main_process and decoder is not None:
                unet.eval()
                
                validation_images = generate_validation_images(
                    text_encoder, unet, vae, args, accelerator, weight_dtype
                )
                
                global_acc = calculate_watermark_accuracy(validation_images, decoder, vae, GT_secret, torch.float32, accelerator)
                Trigger_acc = global_acc
                
                print(f"Step {global_step}: Global Watermark Acc: {global_acc:.4f}")

                validation_dir = os.path.join(args.output_dir, "validation_images")
                os.makedirs(os.path.join(validation_dir, f"watermarked"), exist_ok=True)
                
                for i, img in enumerate(validation_images):
                    img.save(os.path.join(validation_dir, f"watermarked/img_{i}_{global_step}.png"))
                
                unet.train()
            
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
            with torch.no_grad():
                model_input = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            
            timesteps = torch.randint(args.loss_t_min, args.loss_t_max, (bsz,), device=model_input.device)

            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=model_input.device)
            sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)
            noisy_model_input = sqrt_alpha_prod * model_input + sqrt_one_minus_alpha_prod * noise

            encoder_hidden_states = encode_prompt(text_encoder, batch['input_ids']).to(weight_dtype)

            _unet = accelerator.unwrap_model(unet)
            model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states.to(_unet.dtype), return_dict=False)[0]

            with torch.no_grad():
                
                model_original_pred = unet_frozen(noisy_model_input, timesteps, encoder_hidden_states.to(next(unet_frozen.parameters()).dtype), return_dict=False)[0]
                
            watermark_target_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)

            preserve_loss_per_sample = F.mse_loss(model_pred, model_original_pred.detach(), reduction="none").mean(dim=[1, 2, 3]) 
            preserve_loss = preserve_loss_per_sample.mean() 
            
            if args.wmLoss_weight > 0 and FIXED_RESIDUAL is not None:
                
                batch_secret = GT_secret.view(1, -1).repeat(bsz, 1) 
                x0_secret_residual = FIXED_RESIDUAL.repeat(bsz, 1, 1, 1) 
                coefficient_beta_t = (sqrt_alpha_prod / sqrt_one_minus_alpha_prod).to(model_original_pred.dtype)
                
                target_modified_all = model_original_pred - coefficient_beta_t * x0_secret_residual.to(model_original_pred.dtype)
                
                watermark_target_loss_per_sample = F.mse_loss(model_pred, target_modified_all, reduction="none").mean(dim=[1, 2, 3])

                wm_loss_weight_tensor = torch.ones_like(timesteps, dtype=torch.float32)
                if args.preserve_only_at_high_t:
                    
                    wm_loss_weight_tensor = (timesteps < args.loss_t_threshold).float()
                    
                weighted_wm_loss = (watermark_target_loss_per_sample * wm_loss_weight_tensor).mean()
                watermark_target_loss = weighted_wm_loss 
            
            else:
                
                weighted_wm_loss = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)

            total_loss = args.wmLoss_weight * weighted_wm_loss + preserve_loss * args.preserve_weight

            loss_tracker.update(watermark_target_loss, preserve_loss, total_loss) 

            if global_step % 100 == 0 and loss_tracker.count > 0 and accelerator.is_main_process:
                loss_tracker.print_average_losses(global_step, Trigger_acc)

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()
            lr_scheduler.step()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=total_loss.item(), 
                lr=lr_scheduler.get_last_lr()[0],
                acc=f"{Trigger_acc:.3f}"
            )

            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(unet)
                sd = unwrapped.state_dict()
                
                try: save_file(sd, TARGET_UNET)
                except Exception: torch.save(sd, TARGET_UNET.replace('.safetensors', '.bin'))
                torch.save(sd, TARGET_UNET.replace('.safetensors', '.bin'))  
                
                resume_dir = os.path.join(args.output_dir, "resume")
                os.makedirs(resume_dir, exist_ok=True)
                torch.save({"optimizer": optimizer.state_dict(), "scheduler": lr_scheduler.state_dict(),
                            "global_step": global_step}, os.path.join(resume_dir, "state.pt"))
                print(f"Step {global_step}: overwrote UNet + resume state")

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        sd = unwrapped.state_dict()
        try: save_file(sd, TARGET_UNET)
        except Exception: torch.save(sd, TARGET_UNET.replace('.safetensors', '.bin'))
        torch.save(sd, TARGET_UNET.replace('.safetensors', '.bin'))
        resume_dir = os.path.join(args.output_dir, "resume")
        os.makedirs(resume_dir, exist_ok=True)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": lr_scheduler.state_dict(),
                    "global_step": args.max_train_steps}, os.path.join(resume_dir, "state.pt"))
        target = os.path.join(rsv_unet_dir, "pytorch_model.safetensors")
        if os.path.exists(latest):
            import shutil
            shutil.copy2(latest, target)
            print(f"Final overwrite {target}")
    
    accelerator.end_training()

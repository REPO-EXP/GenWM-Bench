"""
Deep generative attacks for watermark robustness evaluation.

Includes:
  - DDIMInversion: Deterministic DDIM inversion → reconstruction
  - DiffInpainting: Random region masking + diffusion regeneration
  - InstructPix2Pix: Text-instruction-based semantic editing

These attack the watermark in latent space by exploiting the diffusion
model's prior to "smooth out" imperceptible signals during generative
reconstruction.
"""

import torch
import gc
import numpy as np
from PIL import Image, ImageDraw
from typing import List

from src.core.interfaces import BaseAttack
from src.core.registry import ATTACKS
from src.core.paths import resolve_model_path

@ATTACKS.register("DDIMInversion")
class DDIMInversionAttack(BaseAttack):
    """
    Deterministic DDIM inversion + reconstruction.

    Pushes an image through DDIM's latent ODE forward (inversion) then
    backward (denoising).  Though mathematically invertible, the finite-step
    approximation loses high frequencies — exactly where watermarks live.

    Parameters
    ----------
    model_id : str
        SD model path (relative to project root or absolute).
    num_steps : int
        Number of inversion / reconstruction steps.
        **Fewer steps = more watermark destruction** (worse reconstruction).
        Recommended range: 10-50.
    """
    def __init__(self,
                 model_id: str = "../../model/stable-diffusion-v1-5",
                 num_steps: int = 25,
                 device: str = 'cuda'):
        self.model_id = resolve_model_path(model_id)
        self.num_steps = int(num_steps)
        self.device = device
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is not None:
            return self._pipe

        from diffusers import StableDiffusionPipeline

        print(f"   [DDIMInv] Loading {self.model_id} …")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    @torch.no_grad()
    def apply(self, image: Image.Image) -> Image.Image:
        from diffusers import DDIMScheduler, DDIMInverseScheduler

        pipe = self._get_pipe()
        if image.mode != 'RGB':
            image = image.convert('RGB')

        orig_size = image.size
        device = self.device

        img_512 = image.resize((512, 512))
        arr = np.array(img_512).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device, dtype=pipe.vae.dtype)
        tensor = 2.0 * tensor - 1.0                     

        latent_dist = pipe.vae.encode(tensor).latent_dist
        latents = latent_dist.sample().to(dtype=pipe.unet.dtype)
        latents = latents * pipe.vae.config.scaling_factor

        tok = pipe.tokenizer([""],
                             padding="max_length",
                             max_length=pipe.tokenizer.model_max_length,
                             truncation=True,
                             return_tensors="pt")
        prompt_embeds = pipe.text_encoder(
            tok.input_ids.to(device))[0].to(dtype=pipe.unet.dtype)

        inv_sched = DDIMInverseScheduler.from_config(
            pipe.scheduler.config)
        inv_sched.set_timesteps(self.num_steps)

        for t in inv_sched.timesteps:
            latent_in = inv_sched.scale_model_input(latents, t)
            noise_pred = pipe.unet(
                latent_in, t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            latents = inv_sched.step(
                noise_pred, t, latents).prev_sample

        fwd_sched = DDIMScheduler.from_config(pipe.scheduler.config)
        fwd_sched.set_timesteps(self.num_steps)

        for t in fwd_sched.timesteps:
            latent_in = fwd_sched.scale_model_input(latents, t)
            noise_pred = pipe.unet(
                latent_in, t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            latents = fwd_sched.step(
                noise_pred, t, latents).prev_sample

        latents = latents / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(latents).sample
        decoded = (decoded / 2.0 + 0.5).clamp(0, 1)

        out_np = decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        out_np = (out_np * 255).astype(np.uint8)
        result = Image.fromarray(out_np).resize(orig_size, Image.LANCZOS)
        return result

    def get_param_str(self) -> str:
        return f"DDIMInv_s{self.num_steps}"

    def _cleanup(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            gc.collect()
            torch.cuda.empty_cache()

@ATTACKS.register("DiffInpainting")
class DiffusionInpaintingAttack(BaseAttack):
    """
    Random region regeneration via diffusion-based inpainting.

    Masks random blocks / grid cells then regenerates the masked area
    with the diffusion model, breaking watermark spatial coherence.

    Two modes:
      - "blend": img2img whole image, blend masked regions (no extra model needed)
      - "inpaint": proper inpainting pipeline (needs runwayml/stable-diffusion-inpainting)

    Parameters
    ----------
    model_id : str
        SD model path. For mode="inpaint" use the inpainting checkpoint.
    mask_ratio : float
        Fraction of image to regenerate (0.0–1.0).  Default 0.3.
    mask_type : str
        "blocks" – random rectangular blocks
        "grid"   – checkerboard-like grid cells
        "holes"  – random circular holes
    num_steps : int
        Diffusion steps for regeneration.
    strength : float
        How strongly to regenerate (0 = original, 1 = pure generation).
    mode : str
        "blend" (default) or "inpaint".
    """
    def __init__(self,
                 model_id: str = "../../model/stable-diffusion-v1-5",
                 mask_ratio: float = 0.3,
                 mask_type: str = "blocks",
                 num_steps: int = 30,
                 strength: float = 0.6,
                 mode: str = "blend",
                 device: str = 'cuda'):
        self.model_id = resolve_model_path(model_id)
        self.mask_ratio = float(mask_ratio)
        self.mask_type = mask_type
        self.num_steps = int(num_steps)
        self.strength = float(strength)
        self.mode = mode
        self.device = device
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is not None:
            return self._pipe

        if self.mode == "inpaint":
            from diffusers import StableDiffusionInpaintPipeline
            print(f"   [DiffInpaint] Loading inpainting model {self.model_id} …")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)
        else:
            from diffusers import StableDiffusionImg2ImgPipeline
            print(f"   [DiffInpaint] Loading img2img model {self.model_id} (blend mode) …")
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)

        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def _make_mask(self, size):
        """Create a binary (0/255) PIL mask."""
        w, h = size
        mask = Image.new("L", size, 0)

        if self.mask_type == "blocks":
            
            draw = ImageDraw.Draw(mask)
            covered = 0.0
            max_attempts = 50
            for _ in range(max_attempts):
                if covered >= self.mask_ratio:
                    break
                bw = np.random.randint(w // 8, w // 3)
                bh = np.random.randint(h // 8, h // 3)
                x1 = np.random.randint(0, w - bw)
                y1 = np.random.randint(0, h - bh)
                draw.rectangle([x1, y1, x1 + bw, y1 + bh], fill=255)
                covered += (bw * bh) / (w * h)

        elif self.mask_type == "grid":
            
            cell_w = max(8, int(w * self.mask_ratio * 2))
            cell_h = max(8, int(h * self.mask_ratio * 2))
            draw = ImageDraw.Draw(mask)
            for y in range(0, h, cell_h):
                for x in range(0, w, cell_w):
                    if np.random.random() < self.mask_ratio:
                        draw.rectangle(
                            [x, y, min(x + cell_w, w), min(y + cell_h, h)],
                            fill=255)

        elif self.mask_type == "holes":
            draw = ImageDraw.Draw(mask)
            radius = max(8, int(min(w, h) * self.mask_ratio * 0.5))
            n_holes = max(1, int(self.mask_ratio * w * h / (np.pi * radius**2)))
            for _ in range(n_holes):
                cx = np.random.randint(radius, w - radius)
                cy = np.random.randint(radius, h - radius)
                draw.ellipse([cx - radius, cy - radius,
                              cx + radius, cy + radius], fill=255)

        return mask

    @torch.no_grad()
    def apply(self, image: Image.Image) -> Image.Image:
        pipe = self._get_pipe()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if pipe is None:
            return image

        try:
            mask = self._make_mask(image.size)

            if self.mode == "inpaint":
                
                result = pipe(
                    image=image,
                    mask_image=mask,
                    prompt="",
                    num_inference_steps=self.num_steps,
                    strength=self.strength,
                    guidance_scale=1.0,
                ).images[0]
            else:
                
                regenerated = pipe(
                    prompt="",
                    image=image,
                    strength=self.strength,
                    num_inference_steps=self.num_steps,
                    guidance_scale=1.0,
                ).images[0]
                result = Image.composite(image, regenerated, mask)

            return result

        except Exception as e:
            print(f"   [DiffInpaint] Error: {e}")
            return image

        finally:
            self._cleanup()

    def get_param_str(self) -> str:
        return (f"DiffInpaint_{self.mask_type}"
                f"_r{self.mask_ratio}_s{self.strength}")

    def _cleanup(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            gc.collect()
            torch.cuda.empty_cache()

@ATTACKS.register("InstructPix2Pix")
class InstructPix2PixAttack(BaseAttack):
    """
    Text-instruction-based image editing via InstructPix2Pix.

    Uses a conditional diffusion model to follow an editing instruction
    (e.g. "change the color to blue"), which semantically alters the
    image and naturally disrupts watermark patterns.

    Parameters
    ----------
    model_id : str
        InstructPix2Pix model. Defaults to "timbrooks/instruct-pix2pix"
        on HuggingFace (download on first use).
    prompt : str
        Editing instruction, e.g. "make it look like a painting".
    image_guidance : float
        How much to preserve the original image (1.0–2.5, default 1.5).
    text_guidance : float
        How strongly to follow the text instruction (default 7.5).
    num_steps : int
        Number of diffusion steps (default 50).
    seed : int
        Random seed for reproducibility.
    """
    
    DEFAULT_PROMPTS = [
        "make it look like a watercolor painting",
        "add film grain and slight blur",
        "change the lighting to dim evening",
        "apply a vintage sepia tone",
        "make the colors slightly muted",
        "add subtle noise and texture",
        "make it look like an old photograph",
        "convert to a pencil sketch style",
    ]

    def __init__(self,
                 model_id: str = "data/models/instruct-pix2pix",
                 prompt: str = "",
                 image_guidance: float = 1.5,
                 text_guidance: float = 7.5,
                 num_steps: int = 50,
                 seed: int = 42,
                 device: str = 'cuda'):
        self.model_id = resolve_model_path(model_id)
        self.prompt = prompt
        self.image_guidance = float(image_guidance)
        self.text_guidance = float(text_guidance)
        self.num_steps = int(num_steps)
        self.seed = int(seed)
        self.device = device
        self._pipe = None
        self._call_count = 0

    def _get_pipe(self):
        if self._pipe is not None:
            return self._pipe

        from diffusers import StableDiffusionInstructPix2PixPipeline

        print(f"   [InstructP2P] Loading {self.model_id} …")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    @torch.no_grad()
    def apply(self, image: Image.Image) -> Image.Image:
        pipe = self._get_pipe()
        if pipe is None:
            return image
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.prompt:
            instruction = self.prompt
        else:
            idx = self._call_count % len(self.DEFAULT_PROMPTS)
            instruction = self.DEFAULT_PROMPTS[idx]
        self._call_count += 1

        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        try:
            result = pipe(
                prompt=instruction,
                image=image,
                image_guidance_scale=self.image_guidance,
                guidance_scale=self.text_guidance,
                num_inference_steps=self.num_steps,
                generator=generator,
            ).images[0]
            return result

        except Exception as e:
            print(f"   [InstructP2P] Error: {e}")
            return image

        finally:
            self._cleanup()

    def get_param_str(self) -> str:
        prompt_short = self.prompt[:20].replace(" ", "_") if self.prompt else "auto"
        return f"InstructP2P_{prompt_short}_ig{self.image_guidance}"

    def _cleanup(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            gc.collect()
            torch.cuda.empty_cache()

@ATTACKS.register("ControlNetInpainting")
class ControlNetInpaintingAttack(BaseAttack):
    def __init__(self,
                 base_model_id: str = "../../model/stable-diffusion-v1-5",
                 controlnet_id: str = "data/models/control_v11p_sd15_inpaint",
                 prompt: str = "",
                 mask_ratio: float = 0.3,
                 num_steps: int = 50,
                 device: str = 'cuda'):
        self.base_model_id = resolve_model_path(base_model_id)
        self.controlnet_id = resolve_model_path(controlnet_id)
        self.prompt = prompt
        self.mask_ratio = float(mask_ratio)
        self.num_steps = int(num_steps)
        self.device = device
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is not None:
            return self._pipe
        from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
        print(f"   [CtrlNetInpaint] Loading controlnet: {self.controlnet_id}")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False).to(self.device)
        print(f"   [CtrlNetInpaint] Loading base model: {self.base_model_id}")
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.base_model_id, controlnet=controlnet, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False).to(self.device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def _make_mask(self, size):
        from PIL import ImageDraw
        w, h = size
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        covered = 0.0
        for _ in range(50):
            if covered >= self.mask_ratio:
                break
            bw = np.random.randint(w // 8, w // 3)
            bh = np.random.randint(h // 8, h // 3)
            x1 = np.random.randint(0, w - bw)
            y1 = np.random.randint(0, h - bh)
            draw.rectangle([x1, y1, x1 + bw, y1 + bh], fill=255)
            covered += (bw * bh) / (w * h)
        return mask

    @staticmethod
    def _make_inpaint_condition(image, mask):
        img_arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        mask_arr = np.array(mask.convert("L")).astype(np.float32) / 255.0
        img_arr[mask_arr > 0.5] = -1.0
        img_arr = np.expand_dims(img_arr, 0).transpose(0, 3, 1, 2)
        return torch.from_numpy(img_arr)

    @torch.no_grad()
    def apply(self, image: Image.Image) -> Image.Image:
        pipe = self._get_pipe()
        if pipe is None:
            return image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        try:
            mask = self._make_mask(image.size)
            control = self._make_inpaint_condition(image, mask).to(
                device=self.device, dtype=torch.float16)
            result = pipe(
                prompt=self.prompt or "", image=image, mask_image=mask,
                control_image=control, num_inference_steps=self.num_steps,
                eta=1.0,
                generator=torch.Generator(device=self.device).manual_seed(1),
            ).images[0]
            return result
        except Exception as e:
            print(f"   [CtrlNetInpaint] Error: {e}")
            return image
        finally:
            self._cleanup()

    def get_param_str(self) -> str:
        p = self.prompt[:15].replace(" ", "_") if self.prompt else "uncond"
        return f"CtrlInpaint_{p}_r{self.mask_ratio}"

    def _cleanup(self):
        if self._pipe is not None:
            del self._pipe; self._pipe = None
            gc.collect(); torch.cuda.empty_cache()

@ATTACKS.register("SVDI2V")
class SVDImage2VideoAttack(BaseAttack):
    def __init__(self,
                 model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
                 num_frames: int = 14,
                 frame_idx: int = -1,
                 num_steps: int = 25,
                 motion_bucket_id: int = 127,
                 fps_id: int = 6,
                 noise_aug_strength: float = 0.02,
                 seed: int = 23,
                 device: str = 'cuda'):
        self.model_id = model_id
        self.num_frames = int(num_frames)
        self.frame_idx = int(frame_idx) if frame_idx >= 0 else self.num_frames // 2
        self.num_steps = int(num_steps)
        self.motion_bucket_id = int(motion_bucket_id)
        self.fps_id = int(fps_id)
        self.noise_aug_strength = float(noise_aug_strength)
        self.seed = int(seed)
        self.device = device
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is not None:
            return self._pipe
        from diffusers import StableVideoDiffusionPipeline
        print(f"   [SVD-I2V] Loading {self.model_id} ...")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    @torch.no_grad()
    def apply(self, image: Image.Image) -> Image.Image:
        pipe = self._get_pipe()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        w, h = image.size
        input_img = image.resize(((w // 64) * 64, (h // 64) * 64))
        try:
            frames = pipe(
                image=input_img, num_frames=self.num_frames,
                num_inference_steps=self.num_steps,
                motion_bucket_id=self.motion_bucket_id, fps_id=self.fps_id,
                noise_aug_strength=self.noise_aug_strength, decode_chunk_size=4,
                generator=torch.Generator(device="cpu").manual_seed(self.seed),
            ).frames[0]
            return frames[min(self.frame_idx, len(frames) - 1)]
        except Exception as e:
            print(f"   [SVD-I2V] Error: {e}")
            import traceback
            traceback.print_exc()
            return image
        finally:
            self._cleanup()

    def get_param_str(self) -> str:
        return f"SVD_f{self.num_frames}_s{self.frame_idx}_m{self.motion_bucket_id}"

    def _cleanup(self):
        if self._pipe is not None:
            del self._pipe; self._pipe = None
            gc.collect(); torch.cuda.empty_cache()

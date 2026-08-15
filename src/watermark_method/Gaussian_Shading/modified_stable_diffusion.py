
from typing import Callable, List, Optional, Union, Any, Dict
import copy
import numpy as np
import PIL

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.utils import logging, BaseOutput
from torchvision.transforms import ToPILImage

logger = logging.get_logger(__name__)  

class ModifiedStableDiffusionPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    init_latents: Optional[torch.FloatTensor]

class ModifiedStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder=None,
        requires_safety_checker: bool = False,
    ):
        super(ModifiedStableDiffusionPipeline, self).__init__(
                vae,
                text_encoder,
                tokenizer,
                unet,
                scheduler,
                safety_checker,
                feature_extractor,
                image_encoder,
                requires_safety_checker)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        watermarking_gamma: float = None,
        watermarking_delta: float = None,
        watermarking_mask: Optional[torch.BoolTensor] = None,
    ):
        
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.count = 0

        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, 
            device, 
            num_images_per_prompt, 
            do_classifier_free_guidance, 
            negative_prompt
        )

        if do_classifier_free_guidance:
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
        else:
            text_embeddings = prompt_embeds

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        init_latents = copy.deepcopy(latents)

        if watermarking_gamma is not None:
            watermarking_mask = torch.rand(latents.shape, device=device) < watermarking_gamma

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        for i, t in enumerate(timesteps):
                
                if watermarking_mask is not None:
                    
                    latents[watermarking_mask] += watermarking_delta * torch.sign(latents[watermarking_mask])

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, 
            output_type=output_type, 
            do_denormalize=do_denormalize
        )

        if not return_dict:
            return (image, has_nsfw_concept)

        return ModifiedStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, init_latents=init_latents)

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents
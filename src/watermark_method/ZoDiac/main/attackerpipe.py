from typing import Callable, List, Optional, Union
from dataclasses import dataclass
import numpy as np
import PIL

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import BaseOutput

@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]

class ReSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        prompt1_steps: Optional[int] = None,
        prompt2: Optional[str] = None,
        head_start_latents: Optional[Union[torch.FloatTensor, list]] = None,
        head_start_step: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        
        if prompt2 is not None:
            text_embeddings2 = self._encode_prompt(
            prompt2, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        if head_start_latents is None:
            num_channels_latents = self.unet.in_channels
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
        else:
            if type(head_start_latents) == list:
                latents = head_start_latents[-1]
                assert len(head_start_latents) == self.scheduler.config.solver_order
                
            else:
                latents = head_start_latents 
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                if not head_start_step or i >= head_start_step: 
                    
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    if prompt1_steps is None or i < prompt1_steps:
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    else:
                        
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings2).sample

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        image = self.decode_latents(latents)

        has_nsfw_concept = False
        
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
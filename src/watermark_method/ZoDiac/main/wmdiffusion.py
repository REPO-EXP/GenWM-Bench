from typing import Callable, List, Optional, Union, Any, Dict
from functools import partial
import numpy as np
import copy
from dataclasses import dataclass
import PIL

import torch
from torch.utils.checkpoint import checkpoint
from diffusers import StableDiffusionPipeline
from diffusers.utils import BaseOutput

@dataclass
class ModifiedStableDiffusionPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]
    nsfw_content_detected: Optional[List[bool]]
    init_latents: Optional[torch.FloatTensor]

class WatermarkStableDiffusionPipeline(StableDiffusionPipeline):
    
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]

    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
        image_encoder: Optional[Any] = None,
        num_inference_steps: int = 50, 
        dtype: Optional[Any] = None,
        **kwargs
    ):
        
        self._default_num_inference_steps = num_inference_steps

        kwargs.pop("image_encoder", None)

        super(WatermarkStableDiffusionPipeline, self).__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
            **kwargs 
        )
        
        self.register_modules(image_encoder=image_encoder)
    
    def decode_latents_tensor(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    def img_tensor_to_numpy(self, tensor):
        return tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    
    def unet_custom_forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs)
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        
        num_inference_steps: Optional[int] = None, 
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        use_trainable_latents: bool = False,
        init_latents: Optional[torch.FloatTensor] = None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if num_inference_steps is None:
            num_inference_steps = self._default_num_inference_steps

        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )
        
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        if not use_trainable_latents:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                init_latents,
            )
            init_latents = copy.deepcopy(latents)
        else:
            if init_latents is None:
                raise ValueError(f"We must have a initial trainable latents.")
            else:
                latents = init_latents

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if not use_trainable_latents:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                else:
                    noise_pred = checkpoint(self.unet_custom_forward, latent_model_input, t, prompt_embeds, cross_attention_kwargs).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type in ['tensor', 'pt']:
            image = self.decode_latents_tensor(latents) 
            has_nsfw_concept = None
        else:
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            
            do_denormalize = [True] * image.shape[0] if has_nsfw_concept is None else [not has_nsfw for has_nsfw in has_nsfw_concept]
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept, init_latents)

        return ModifiedStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, init_latents=init_latents)

class WMDetectStableDiffusionPipeline(WatermarkStableDiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]

    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
        image_encoder=None,
        num_inference_steps: int = 50,
        dtype: Optional[Any] = None,
    ):
        super(WMDetectStableDiffusionPipeline, self).__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
            image_encoder=image_encoder,
            num_inference_steps=num_inference_steps,
            dtype=dtype
        )
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)

    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings
    
    @torch.inference_mode()
    def get_image_latents(self, image: torch.Tensor, sample=True, rng_generator=None):
        image = 2.0 * image - 1.0
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * self.vae.config.scaling_factor
        return latents

    def backward_ddim(self, x_t, alpha_t, alpha_tm1, eps_xt):
        return (
            alpha_tm1**0.5
            * (
                (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
                + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
            )
            + x_t
        )
    
    @torch.inference_mode()
    def backward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        
        num_inference_steps: Optional[int] = None, 
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: True = False,
        **kwargs,
    ):
        
        if num_inference_steps is None:
            num_inference_steps = self._default_num_inference_steps

        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        
        if latents is not None:
             latents = latents * self.scheduler.init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False

        for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            prev_timestep = (
                t
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps
            )
            
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
                
            latents = self.backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
        return latents
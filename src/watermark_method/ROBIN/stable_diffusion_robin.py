
from typing import Callable, List, Optional, Union, Any, Dict
import copy
import os
import numpy as np
import PIL
from statistics import mean
from tqdm import tqdm
import itertools
import time

from torch import inference_mode
import torch
from torch.cuda.amp import GradScaler, autocast
from diffusers import StableDiffusionPipeline
from diffusers.utils import BaseOutput
import logging
from optim_utils import *

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler('logs/output.log', mode='a', encoding='utf-8')  
    ]
)

logger = logging.getLogger(__name__)  

def get_pred_ori(x_t, alpha_t, epx_xt):
    
    beta_t = 1 - alpha_t
    return (
        (x_t - beta_t ** (0.5)
         * epx_xt) / alpha_t ** (0.5)
    )

class ROBINStableDiffusionPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    init_latents: Optional[torch.FloatTensor]
    latents: Optional[torch.FloatTensor]
    inner_latents: Optional[List[torch.FloatTensor]]

class ROBINStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
        **kwargs  
    ):
        
        super(ROBINStableDiffusionPipeline, self).__init__(
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
        watermarking_mask: Optional[torch.BoolTensor] = None,
        watermarking_steps: int = None,
        args = None,
        gt_patch = None,
        lguidance = None,
        opt_acond = None
    ):
        
        inner_latents = []
        
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

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

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        inner_latents.append(init_latents)

        max_train_steps=1  
        latents_wm = None
        text_embeddings_opt = None
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        start_time = time.time()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if (watermarking_steps is not None) and (i >= watermarking_steps):
                    mask = watermarking_mask  
                    if i == watermarking_steps:
                        latents_wm = inject_watermark(latents, mask,gt_patch, args)  
                        inner_latents[-1] = latents_wm  
                        if opt_acond is not None:
                            uncond, cond = text_embeddings.chunk(2)
                            opt_acond = opt_acond.to(cond.dtype)
                            text_embeddings_opt = torch.cat([uncond, opt_acond, cond])  
                        else:
                            text_embeddings_opt = text_embeddings.clone()
                        if lguidance is not None:
                            guidance_scale = lguidance  

                    latents_wm, _ = self.xn1_latents_3(latents_wm,do_classifier_free_guidance,t
                                                            ,text_embeddings_opt,guidance_scale,**extra_step_kwargs)

                if (watermarking_steps is None) or (watermarking_steps is not None and i < watermarking_steps):
                    latents, _ = self.xn1_latents(latents,do_classifier_free_guidance,t
                                                            ,text_embeddings,guidance_scale,**extra_step_kwargs)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                
                if (watermarking_steps is not None and i < watermarking_steps) or (watermarking_steps is None):
                    inner_latents.append(latents)   
                else: 
                    inner_latents.append(latents_wm)

                if watermarking_steps is not None and watermarking_steps == 50:
                    latents_wm = inject_watermark(latents, watermarking_mask,gt_patch, args)  
                    inner_latents[-1] = latents_wm  

        end_time = time.time()
        execution_time = end_time - start_time
        
        latents_to_decode = latents_wm if latents_wm is not None else latents
        
        image = self.vae.decode(latents_to_decode / self.vae.config.scaling_factor, return_dict=False)[0]

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
            
        if text_embeddings_opt is not None:
            return ROBINStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, init_latents=init_latents, latents=latents, inner_latents=inner_latents,gt_patch=gt_patch,opt_acond=text_embeddings_opt[0],time=execution_time)
        else:
            return ROBINStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, init_latents=init_latents, latents=latents, inner_latents=inner_latents,gt_patch=gt_patch,time=execution_time)

    def optimizer_wm_prompt(self, dataloader,hyperparameters, mask,opt_wm,save_path,args,
                            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                            eta: float = 0.0,     ):
        train_batch_size = hyperparameters["train_batch_size"]
        gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
        learning_rate = hyperparameters["learning_rate"]
        max_train_steps = hyperparameters["max_train_steps"]
        output_dir = hyperparameters["output_dir"]
        gradient_checkpointing = hyperparameters["gradient_checkpointing"]

        text_encoder = self.text_encoder
        unet = self.unet
        vae = self.vae
        scheduler = self.scheduler

        freeze_params(vae.parameters())
        freeze_params(unet.parameters())
        freeze_params(text_encoder.parameters())

        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=hyperparameters["mixed_precision"]
        )

        if gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            unet.enable_gradient_checkpointing()

        if hyperparameters["scale_lr"]:
            learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
            )

        tester_prompt = '' 
        text_embeddings = self.get_text_embedding(tester_prompt)  

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        unet, text_encoder, dataloader,text_embeddings = accelerator.prepare(
            unet, text_encoder, dataloader, text_embeddings
        ) 

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        vae.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)

        vae.eval()
        
        unet.train()

        num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataloader)}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        scaler = GradScaler()
        
        opt_wm_embedding = self.get_text_embedding('')
        null_embedding = opt_wm_embedding.clone()
        import os 
        if args.resume_from_checkpoint is not None and os.path.isfile(args.resume_from_checkpoint):
            logger.info(f"Pipeline: Loading checkpoint from {args.resume_from_checkpoint}...")
            
            loaded_state = torch.load(args.resume_from_checkpoint, map_location=accelerator.device)
            
            if 'opt_acond' in loaded_state:
                
                opt_wm_embedding = loaded_state['opt_acond']
                
                opt_wm_embedding = opt_wm_embedding.to(accelerator.device).detach()
                logger.info(f"Pipeline: Successfully RESUMED 'opt_acond' (Size: {opt_wm_embedding.shape})")
            else:
                logger.warning("Pipeline: 'opt_acond' key not found in checkpoint!")
                
            if 'opt_wm' in loaded_state:
                opt_wm = loaded_state['opt_wm'].to(accelerator.device).detach()
                logger.info(f"Pipeline: Successfully RESUMED 'opt_wm' (Size: {opt_wm.shape})")
            else:
                logger.warning("Pipeline: 'opt_wm' key not found in checkpoint!")

        total_time = 0
        with autocast():
            for epoch in range(num_train_epochs):
                for step, batch in enumerate(dataloader):
                    with accelerator.accumulate(unet):
                        
                        gt_tensor = batch["pixel_values"]
                        image = 2.0 * gt_tensor - 1.0
                        latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample().detach()
                        latents = latents * 0.18215
                       
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        
                        timesteps = torch.randint(200, 300, (bsz,), device=latents.device).long()  

                        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                        opt_wm = opt_wm.to(noisy_latents.device).to(torch.complex64)  
                        
                        init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(noisy_latents), dim=(-1, -2))
                        init_latents_w_fft[mask] = opt_wm[mask].clone()
                        init_latents_w_fft.requires_grad = True
                        noisy_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

                        prompt = batch["prompt"]
                        
                        cond_embedding = self.get_text_embedding(prompt)
                        text_embeddings = torch.cat([opt_wm_embedding, cond_embedding, null_embedding]) 
                        text_embeddings.requires_grad = True

                        latent_model_input = torch.cat([noisy_latents] * 3)
                        latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps)
                        noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings).sample
                        noise_pred_wm, noise_pred_text, noise_pred_null = noise_pred.chunk(3)
                        noise_pred = noise_pred_null + 3.5 * (noise_pred_text - noise_pred_null) + 3.5 * (noise_pred_wm - noise_pred_null)   
                        
                        current_timestep = timesteps.item()
                        sigma = self.scheduler.sigmas[self.scheduler.timesteps.tolist().index(current_timestep)]
                        alpha_t, sigma_t = self.scheduler._sigma_to_alpha_sigma_t(sigma)
                        alpha_t = alpha_t.item() if hasattr(alpha_t, 'item') else alpha_t
                        sigma_t = sigma_t.item() if hasattr(sigma_t, 'item') else sigma_t

                        x0_latents = (noisy_latents - sigma_t * noise_pred) / alpha_t

                        x0_tensor = self.decode_latents_wgrad(x0_latents)

                        loss_noise = F.mse_loss(x0_tensor.float(), gt_tensor.float(), reduction="mean")  
                        loss_wm = torch.mean(torch.abs(opt_wm[mask].real))
                        loss_constrain = F.mse_loss(noise_pred_wm.float(), noise_pred_null.float(), reduction="mean")  

                        if (global_step // 500) % 2 == 0:
                            loss = 10 * loss_noise + loss_constrain - 0.00001 * loss_wm  
                            accelerator.backward(loss)
                            with torch.no_grad():  
                                grads = init_latents_w_fft.grad
                                init_latents_w_fft = init_latents_w_fft - 1.0 * grads  
                                init_latents_w_fft = to_ring(init_latents_w_fft, args)
                                opt_wm = init_latents_w_fft.detach()
                        else:
                            loss = 10 * loss_noise + loss_constrain  
                            accelerator.backward(loss)
                            with torch.no_grad():  
                                grads = text_embeddings.grad
                                text_embeddings = text_embeddings - 5e-04 * grads  
                                opt_wm_embedding = text_embeddings[0].unsqueeze(0).detach()  

                        print(f'global_step: {global_step}, loss_mse: {loss_noise}, loss_wm: {loss_wm}, loss_cons: {loss_constrain},loss: {loss}')

                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        if global_step % hyperparameters["save_steps"] == 0:
                            import os
                            os.makedirs(save_path, exist_ok=True)
                            path = os.path.join(save_path, f"optimized_wm5-30_embedding-step-{global_step}.pt")
                            torch.save({'opt_acond': opt_wm_embedding, 'opt_wm': opt_wm.cpu()}, path)

                    logs = {"loss": loss.detach().item()}
                    progress_bar.set_postfix(**logs)

                    if global_step >= max_train_steps:
                        break

                accelerator.wait_for_everyone()

        return opt_wm, opt_wm_embedding

    def xn1_latents(self,latents,do_classifier_free_guidance,t
                        ,text_embeddings,guidance_scale,**extra_step_kwargs):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        return latents, noise_pred
    
    def xn1_latents_3(self,latents,do_classifier_free_guidance,t
                        ,text_embeddings,guidance_scale,**extra_step_kwargs):
        latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text1, noise_pred_text2 = noise_pred.chunk(3)
            noise_pred = noise_pred_uncond + 3.5 * (noise_pred_text1 - noise_pred_uncond) + 3.5 * (noise_pred_text2 - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        return latents, noise_pred
    
    def decode_latents_wgrad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def get_noise(
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
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            watermarking_mask: Optional[torch.BoolTensor] = None,
            watermarking_steps: int = None,
            args = None,
            gt_patch = None,
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

            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            inner_noise = []
            uncond_noise = []
            textcond_noise = []
            guidance_noise = []
            freq_noise = []
            noise_sim = []
            uncond_sim = []
            textcond_sim = []
            guidance_sim = []
            freq_sim = []
            latents_sim = []
            noise_sim_wm = []
            uncond_sim_wm = []
            textcond_sim_wm = []
            guidance_sim_wm = []
            freq_sim_wm = []
            latents_sim_wm = []
            mask = watermarking_mask
            latents_wm = None

            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with torch.no_grad(): 
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        
                        if (watermarking_steps is not None) and (i == watermarking_steps):
                            latents_wm = inject_watermark(latents, mask,gt_patch, args)

                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        inner_noise.append(torch.abs(noise_pred).mean().item())
                        uncond_noise.append(torch.abs(noise_pred_uncond).mean().item())
                        textcond_noise.append(torch.abs(noise_pred_text).mean().item())
                        guidance_noise.append(torch.abs(noise_pred - noise_pred_uncond).mean().item())
                        freq_noise.append(torch.abs(torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2)).real).mean().item())

                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                        if i != 0:
                            noise_sim.append(cosine_distance(prev_noise, noise_pred).mean().item())
                            uncond_sim.append(cosine_distance(prev_uncond, noise_pred_uncond).mean().item())
                            textcond_sim.append(cosine_distance(prev_cond, noise_pred_text).mean().item())
                            guidance_sim.append(cosine_distance(prev_guidance, noise_pred - noise_pred_uncond).mean().item())
                            freq_sim.append(fcosine_distance(prev_latents, latents).mean().item())
                            latents_sim.append(cosine_distance(prev_latents, latents).mean().item())

                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and i % callback_steps == 0:
                                callback(i, t, latents)

                        if latents_wm is not None:
                            
                            latent_model_input_wm = torch.cat([latents_wm] * 2) if do_classifier_free_guidance else latents_wm

                            latent_model_input_wm = self.scheduler.scale_model_input(latent_model_input_wm, t)

                            noise_pred_wm = self.unet(latent_model_input_wm, t, encoder_hidden_states=text_embeddings).sample

                            if do_classifier_free_guidance:
                                noise_pred_uncond_wm, noise_pred_text_wm = noise_pred_wm.chunk(2)
                                noise_pred_wm = noise_pred_uncond_wm + guidance_scale * (noise_pred_text_wm - noise_pred_uncond_wm)  
                                
                            noise_sim_wm.append(cosine_distance(noise_pred_wm, noise_pred).mean().item())
                            uncond_sim_wm.append(cosine_distance(noise_pred_uncond_wm, noise_pred_uncond).mean().item())
                            textcond_sim_wm.append(cosine_distance(noise_pred_text_wm, noise_pred_text).mean().item())
                            guidance_sim_wm.append(cosine_distance(noise_pred_wm-noise_pred_uncond_wm, noise_pred-noise_pred_uncond).mean().item())
                            latents_sim_wm.append(cosine_distance(latents_wm, latents).mean().item())
                            freq_sim_wm.append(fcosine_distance(latents_wm, latents).mean().item())
       
                            latents_wm = self.scheduler.step(noise_pred_wm, t, latents_wm, **extra_step_kwargs).prev_sample

                        prev_noise = noise_pred
                        prev_uncond = noise_pred_uncond
                        prev_cond = noise_pred_text
                        prev_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)
                        prev_latents = latents

            if latents_wm is None:
                return inner_noise,uncond_noise,textcond_noise,guidance_noise,freq_noise,noise_sim,uncond_sim,textcond_sim,guidance_sim,freq_sim
            else:
                return inner_noise,uncond_noise,textcond_noise,guidance_noise,freq_noise,noise_sim,uncond_sim,textcond_sim,guidance_sim,latents_sim, freq_sim, noise_sim_wm, uncond_sim_wm, textcond_sim_wm, guidance_sim_wm, latents_sim_wm, freq_sim_wm

    @torch.no_grad()
    def get_watermark_persistence(
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
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            watermarking_mask: Optional[torch.BoolTensor] = None,
            watermarking_steps: int = None,
            args = None,
            gt_patch = None,
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

            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            wm_nmse = []
            mask = watermarking_mask

            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with torch.no_grad(): 
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        
                        if (watermarking_steps is not None) and (i == watermarking_steps):
                            latents = inject_watermark(latents, mask,gt_patch, args)

                        if (watermarking_steps is not None)and (i >= watermarking_steps):
                            freq_latents = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2)).real 
                            wm_nmse.append(error_nmse(gt_patch[mask].real, freq_latents[mask]))

                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and i % callback_steps == 0:
                                callback(i, t, latents)

            freq_latents = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2)).real 
            wm_nmse.append(error_nmse(gt_patch[mask].real, freq_latents[mask]))
            return wm_nmse

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
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
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

import math
import numpy as np
import scipy
from PIL import Image
import torch
import torchvision.transforms as tforms
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
from diffusers.models import AutoencoderKL

def circle_mask(size=128, r=16, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0)**2 + (y-y0)**2)<= r**2

def get_pattern(shape,pipe,batchsize,w_seed=999999):
    g = torch.Generator(device=pipe.device)
    g.manual_seed(w_seed)
    gt_init = pipe.prepare_latents(batchsize, pipe.unet.config.in_channels,
                                   512, 512,
                                   pipe.unet.dtype, pipe.device, g)
    gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
    
    gt_patch_tmp = gt_patch.clone().detach()
    for i in range(shape[-1] // 2, 0, -1):
        tmp_mask = circle_mask(gt_init.shape[-1], r=i)
        tmp_mask = torch.tensor(tmp_mask)
        for j in range(gt_patch.shape[1]):
            gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch

def transform_img(image):
    tform = tforms.Compose([tforms.Resize(512),tforms.CenterCrop(512),tforms.ToTensor()])
    image = tform(image)
    return 2.0 * image - 1.0

def get_noise(pipe, w_mask, w_key, batch_size, generator=None):
    
    init_latents = pipe.prepare_latents(
        batch_size, 
        pipe.unet.config.in_channels, 
        512, 512, 
        pipe.unet.dtype, 
        pipe.device, 
        generator
    )
    
    init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
    init_latents_fft[w_mask] = w_key[w_mask].clone()
    init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real
    
    init_latents[init_latents == float("Inf")] = 3
    init_latents[init_latents == float("-Inf")] = -3

    return init_latents

def detect(image,pipe,w_key,w_mask,alpha=0.01):
    
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    img = transform_img(image).unsqueeze(0).to(pipe.unet.dtype).to(pipe.device)
    image_latents = pipe.vae.encode(img).latent_dist.mode() * 0.13025
    inverted_latents = pipe(prompt="", latents=image_latents, guidance_scale=1, num_inference_steps=25, output_type="latent")
    inverted_latents = inverted_latents.images

    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))[w_mask].flatten()
    target = w_key[w_mask].flatten()
    inverted_latents_fft = torch.concatenate([inverted_latents_fft.real, inverted_latents_fft.imag])
    target = torch.concatenate([target.real, target.imag])

    sigma = inverted_latents_fft.std()
    lamda = (target ** 2 / sigma ** 2).sum().item()
    x = (((inverted_latents_fft - target) / sigma) ** 2).sum().item()
    p_value = scipy.stats.ncx2.cdf(x=x, df=len(target), nc=lamda)
    
    pipe.scheduler = curr_scheduler
    return max(0.0, 1-1/math.log(5/p_value,10)),(p_value <= alpha)
    
def generate(prompt,pipe):
    return pipe(prompt=prompt, num_inference_steps=50, latents=get_noise()).images[0]
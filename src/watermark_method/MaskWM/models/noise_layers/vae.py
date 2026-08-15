import random
import torch
from torch import nn

try:
    from diffusers import AutoencoderKL
    _DIFFUSERS_AVAILABLE = True
except Exception:
    _DIFFUSERS_AVAILABLE = False

try:
    from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor
    _COMPRESSAI_AVAILABLE = True
except Exception:
    _COMPRESSAI_AVAILABLE = False

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        if not (_DIFFUSERS_AVAILABLE and _COMPRESSAI_AVAILABLE):
            raise ImportError("[MaskWM] VAE noise layer requires a working `diffusers` AutoencoderKL "
                               "and `compressai` (pip install compressai). "
                               "Not needed for plain inference (default noise_layers='Identity()').")
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", local_files_only=True, subfolder="vae", torch_dtype=torch.float16
        ).eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.bmshj2018 = bmshj2018_hyperprior(quality=5, pretrained=True).eval()
        for p in self.bmshj2018.parameters():
            p.requires_grad = False

        self.cheng2020 = cheng2020_anchor(quality=5, pretrained=True).eval()
        for p in self.cheng2020.parameters():
            p.requires_grad = False

    def forward(self, image, mask):
        vae_type = random.choice(["vae", "bmshj2018", "cheng2020"])
        
        if vae_type == "vae":
            latents = self.vae.encode(image.half()).latent_dist.mode()
            noised_image = self.vae.decode(latents, return_dict=False)[0].float()
        elif vae_type == "bmshj2018":
            out_net = self.bmshj2018(torch.clamp((image + 1) / 2, 0, 1))
            out_net['x_hat'].clamp_(0, 1)
            noised_image = 2 * out_net['x_hat'] - 1
        else:
            out_net = self.cheng2020(torch.clamp((image + 1) / 2, 0, 1))
            out_net['x_hat'].clamp_(0, 1)
            noised_image = 2 * out_net['x_hat'] - 1
        return noised_image, mask

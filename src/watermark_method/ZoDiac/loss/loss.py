import logging
import math
import os as _os
import shutil as _shutil
import torch
import torch.nn as nn
from torchvision.models import alexnet as _alexnet_builder, AlexNet_Weights

_CACHE_DIR = _os.path.expanduser('~/.cache/torch/hub/checkpoints')
_CACHE_FILE = _os.path.join(_CACHE_DIR, 'alexnet-owt-7be5be79.pth')

if not _os.path.exists(_CACHE_FILE):
    
    _proj_root = _os.path.abspath(_os.path.join(
        _os.path.dirname(__file__), '..', '..', '..', '..'))
    _local = _os.path.join(_proj_root, 'data', 'models', 'Alexnet', 'alexnet-owt-7be5be79.pth')
    if _os.path.exists(_local):
        _os.makedirs(_CACHE_DIR, exist_ok=True)
        _shutil.copy(_local, _CACHE_FILE)

if not _os.path.exists(_CACHE_FILE):
    _proj_root = _os.path.abspath(_os.path.join(
        _os.path.dirname(__file__), '..', '..', '..', '..'))
    _local = _os.path.join(_proj_root, 'data', 'models', 'Alexnet', 'alexnet-owt-7be5be79.pth')
    _CACHE_FILE = _local  

_alexnet = _alexnet_builder(weights=None)
_alexnet.load_state_dict(torch.load(_CACHE_FILE, map_location='cpu'))
_alexnet.eval()

import torchvision.models as _tv_models
_orig_alexnet = _tv_models.alexnet
def _patched_alexnet(*, weights=None, progress=True, **kwargs):
    if weights is not None or kwargs.pop('pretrained', False):
        return _alexnet
    return _orig_alexnet(weights=None, progress=progress, **kwargs)
_tv_models.alexnet = _patched_alexnet

import lpips
from .pytorch_ssim import SSIM

logger = logging.getLogger()
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class LossProvider(nn.Module):
    def __init__(self, loss_weights, device):
        super(LossProvider, self).__init__()
        self.loss_weights = loss_weights
        self.device = device

        self.loss_img = nn.MSELoss().to(device)  
        self.loss_w = nn.L1Loss().to(device)     
        self.loss_ssim = SSIM().to(device)       

        print("[ZoDiac Loss] Loading LPIPS (AlexNet)...")
        
        self.loss_percep = lpips.LPIPS(net='alex').to(device)
        
        self.loss_per = lambda pred, gt: self.loss_percep((pred * 2 - 1), (gt * 2 - 1)).mean()

    def calculate_psnr(self, img1, img2):
        
        if img1.min() < -0.5: img1 = (img1 + 1) / 2.0
        if img2.min() < -0.5: img2 = (img2 + 1) / 2.0
        
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(1.0 / math.sqrt(mse.item()))

    def __call__(self, pred_img_tensor, gt_img_tensor, init_latents_wm, wm_pipe, print_metrics=False):
        
        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_wm), dim=(-1, -2))
        
        lossW = self.loss_w(init_latents_fft[wm_pipe.watermarking_mask], 
                            wm_pipe.gt_patch[wm_pipe.watermarking_mask]) * self.loss_weights[3]
        
        lossI = self.loss_img(pred_img_tensor, gt_img_tensor) * self.loss_weights[0]
        
        lossP = self.loss_per(pred_img_tensor, gt_img_tensor) * self.loss_weights[1]
        
        lossS = (1 - self.loss_ssim(pred_img_tensor, gt_img_tensor)) * self.loss_weights[2]
        
        loss = lossW + lossI + lossP + lossS
        
        if print_metrics:
            psnr = self.calculate_psnr(pred_img_tensor, gt_img_tensor)
            logging.info(f'Loss - Watermark: {lossW.item():.6f}, Image: {lossI.item():.6f}, '
                         
                         )
            logging.info(f'Metrics - PSNR: {psnr:.2f} dB')
        
        return loss

    def evaluate_metrics(self, pred_img_tensor, gt_img_tensor, init_latents=None, wm_pipe=None):
        
        metrics = {}
        
        metrics['ssim'] = self.loss_ssim(pred_img_tensor, gt_img_tensor).item()
        metrics['psnr'] = self.calculate_psnr(pred_img_tensor, gt_img_tensor)
        metrics['mse'] = torch.mean((pred_img_tensor - gt_img_tensor) ** 2).item()
        
        if init_latents is not None and wm_pipe is not None:
            init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
            
            wm_loss = self.loss_w(init_latents_fft[wm_pipe.watermarking_mask], 
                                  wm_pipe.gt_patch[wm_pipe.watermarking_mask]).item()
            metrics['watermark_loss'] = wm_loss
        
        return metrics
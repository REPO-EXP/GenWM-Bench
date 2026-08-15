import torch
import numpy as np
import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.core.interfaces import BaseMetric
from src.core.registry import METRICS

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_LOCAL_HUB = os.path.join(_PROJ_ROOT, 'data', 'models', 'torch_hub')
os.makedirs(os.path.join(_LOCAL_HUB, 'checkpoints'), exist_ok=True)
torch.hub.set_dir(_LOCAL_HUB)

_ALEX_SRC = os.path.join(_PROJ_ROOT, 'data', 'models', 'Alexnet', 'alexnet-owt-7be5be79.pth')
_ALEX_DST = os.path.join(_LOCAL_HUB, 'checkpoints', 'alexnet-owt-7be5be79.pth')
if os.path.exists(_ALEX_SRC) and not os.path.exists(_ALEX_DST):
    os.symlink(_ALEX_SRC, _ALEX_DST)
import lpips

try:
    from pytorch_msssim import ms_ssim
except ImportError:
    ms_ssim = None
    print("[Warning] pytorch-msssim not installed. MSSIM will be 0.")

@METRICS.register("BasicQualityMetrics")
class BasicQualityMetrics(BaseMetric):
    def __init__(self, config=None, **kwargs):
        if config is None: config = {}
        config.update(kwargs)
        super().__init__(config)
        
        self.enable_lpips = not config.get('disable_lpips', True)

        if self.enable_lpips:
            print(f"[Metrics] Loading LPIPS model on {self.device}...")
            
            self.lpips_model = lpips.LPIPS(net='alex').eval().to(self.device)
        else:
            self.lpips_model = None
            print("[Metrics] LPIPS evaluation is DISABLED.")

    def calculate(self, **kwargs) -> dict:
        img_orig = kwargs.get('img_gen_clean')
        img_wm = kwargs.get('img_gen_wm')
        
        if img_orig is None: img_orig = kwargs.get('img_orig')
        if img_wm is None: img_wm = kwargs.get('img_wm')
        
        if img_orig is None or img_wm is None:
            return {}

        img_orig_np = self._to_numpy(img_orig)
        img_wm_np = self._to_numpy(img_wm)

        psnr_val = self._calc_psnr(img_orig_np, img_wm_np)
        ssim_val = self._calc_ssim(img_orig_np, img_wm_np)

        img_orig_tensor_norm, img_orig_tensor_01 = self._to_tensor_pair(img_orig_np)
        img_wm_tensor_norm, img_wm_tensor_01 = self._to_tensor_pair(img_wm_np)

        if self.enable_lpips:
            lpips_val = self._calc_lpips(img_orig_tensor_norm, img_wm_tensor_norm)
        else:
            lpips_val = 0.0  

        mssim_val = self._calc_mssim(img_orig_tensor_01, img_wm_tensor_01)
        print(f"psnr: {float(psnr_val)},ssim: {float(ssim_val)},mssim: {float(mssim_val)},lpips: {float(lpips_val)}")
        return {
            "psnr": float(psnr_val),
            "ssim": float(ssim_val),
            "mssim": float(mssim_val),
            "lpips": float(lpips_val)
        }

    def _to_numpy(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] in [1, 3]: 
                img = np.transpose(img, (1, 2, 0)) * 255.0
            elif img.ndim == 4: 
                img = np.transpose(img[0], (1, 2, 0)) * 255.0
        if not isinstance(img, np.ndarray): img = np.array(img)
        return img.astype(np.float32)

    def _to_tensor_pair(self, img_np):
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return (tensor / 127.5 - 1.0), (tensor / 255.0)

    def _calc_psnr(self, img1, img2):
        return peak_signal_noise_ratio(img1, img2, data_range=255)

    def _calc_ssim(self, img1, img2):
        try:
            return structural_similarity(img1, img2, data_range=255, channel_axis=2)
        except TypeError:
            return structural_similarity(img1, img2, data_range=255, multichannel=True)

    def _calc_lpips(self, t1, t2):
        with torch.no_grad(): dist = self.lpips_model(t1, t2)
        return dist.item()

    def _calc_mssim(self, t1, t2):
        if ms_ssim is None: return 0.0
        with torch.no_grad(): 
            val = ms_ssim(t1, t2, data_range=1.0, size_average=True)
        return val.item()
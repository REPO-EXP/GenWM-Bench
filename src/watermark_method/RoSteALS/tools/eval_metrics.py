import torch
import numpy as np
import skimage.metrics
import lpips
from PIL import Image
from .sifid import SIFID

def resize_array(x, size=256):
    
    if x.shape[1] != size:
        x = [Image.fromarray(x[i]).resize((size, size), resample=Image.BILINEAR) for i in range(x.shape[0])]
        x = np.array([np.array(i) for i in x])
    return x

def resize_tensor(x, size=256):
    
    if x.shape[2] != size:
        x = torch.nn.functional.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    return x

def normalise(x):
    
    x = x.astype(np.float32)
    x = x / 255
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x)
    x = x.permute(0, 3, 1, 2)
    return x

def unormalise(x, vrange=[-1, 1]):
    
    x = (x - vrange[0])/(vrange[1] - vrange[0])
    x = x * 255
    x = x.permute(0, 2, 3, 1)
    x = x.cpu().numpy().astype(np.uint8)
    return x

def compute_mse(x, y):
    
    return np.square(np.float64(x) - np.float64(y)).reshape(x.shape[0], -1).mean(axis=1)

def compute_psnr(x, y):
    
    return 10 * np.log10(255 ** 2 / compute_mse(x, y))

def compute_ssim(x, y):
    
    return np.array([skimage.metrics.structural_similarity(xi, yi, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255) for xi, yi in zip(x, y)])

def compute_lpips(x, y, net='alex'):
    
    lpips_fn = lpips.LPIPS(net=net, verbose=False).cuda() if isinstance(net, str) else net
    x, y = x.cuda(), y.cuda()
    return lpips_fn(x, y).detach().cpu().numpy().squeeze()

def compute_sifid(x, y, net=None):
    
    fn = SIFID() if net is None else net
    out = [fn(xi, yi) for xi, yi in zip(x, y)]
    return np.array(out)
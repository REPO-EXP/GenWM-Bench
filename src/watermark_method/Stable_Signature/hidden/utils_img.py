
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.transforms import functional
from augly.image import functional as aug_functional

import kornia.augmentation as K

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
image_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unnormalize_rgb = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
normalize_yuv = transforms.Normalize(mean=[0.5, 0, 0], std=[0.5, 1, 1])
unnormalize_yuv = transforms.Normalize(mean=[-0.5/0.5, 0, 0], std=[1/0.5, 1/1, 1/1])

def normalize_img(x):
    
    return (x - image_mean.to(x.device)) / image_std.to(x.device)

def unnormalize_img(x):
    
    return (x * image_std.to(x.device)) + image_mean.to(x.device)

def round_pixel(x):
    
    x_pixel = 255 * unnormalize_img(x)
    y = torch.round(x_pixel).clamp(0, 255)
    y = normalize_img(y/255.0)
    return y

def clamp_pixel(x):
    
    x_pixel = 255 * unnormalize_img(x)
    y = x_pixel.clamp(0, 255)
    y = normalize_img(y/255.0)
    return y

def project_linf(x, y, radius):
    
    delta = x - y
    delta = 255 * (delta * image_std.to(x.device))
    delta = torch.clamp(delta, -radius, radius)
    delta = (delta / 255.0) / image_std.to(x.device)
    return y + delta

def psnr(x, y):
    
    delta = x - y
    delta = 255 * (delta * image_std.to(x.device))
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) 
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2, dim=(1,2,3)))  
    return psnr

def center_crop(x, scale):
    
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]

    x = functional.center_crop(x, new_edges_size)
    return x

def resize(x, scale):
    
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)

def rotate(x, angle):
    
    return functional.rotate(x, angle)

def adjust_brightness(x, brightness_factor):
    
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))

def adjust_contrast(x, contrast_factor):
    
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

def jpeg_compress(x, quality_factor):
    
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    x = unnormalize_img(x)
    for ii,img in enumerate(x):
        pil_img = to_pil(img)
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)

def gaussian_blur(x, sigma=1):
    
    x = unnormalize_img(x)
    x = functional.gaussian_blur(x, sigma=sigma, kernel_size=21)
    x = normalize_img(x)
    return x

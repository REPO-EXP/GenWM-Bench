import torch
import numpy as np
import random
from PIL import Image, ImageFilter
from torchvision import transforms
from io import BytesIO
from src.core.interfaces import BaseAttack
from src.core.registry import ATTACKS
from torchvision.transforms import functional as TF
try:
    import kornia as K
except ImportError:
    pass

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@ATTACKS.register("Contrast")
class ContrastAttack(BaseAttack):
    def __init__(self, contrast=1.0):
        
        self.contrast = contrast

    def apply(self, image: Image.Image) -> Image.Image:
        
        jitter = transforms.ColorJitter(contrast=(self.contrast, self.contrast))
        return jitter(image)

    def get_param_str(self):
        return f"Contrast_{self.contrast}"

@ATTACKS.register("Brightness")
class BrightnessAttack(BaseAttack):
    def __init__(self, brightness=1.0):
        self.brightness = brightness

    def apply(self, image: Image.Image) -> Image.Image:
        
        jitter = transforms.ColorJitter(brightness=(self.brightness, self.brightness))
        return jitter(image)

    def get_param_str(self):
        return f"Bright_{self.brightness}"

@ATTACKS.register("JPEG")
class JPEGAttack(BaseAttack):
    def __init__(self, quality=80):
        self.quality = int(quality)

    def apply(self, image: Image.Image) -> Image.Image:
        if image.mode != 'RGB': image = image.convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)

    def get_param_str(self):
        return f"JPEG_{self.quality}"

@ATTACKS.register("GaussianBlur")
class GaussianBlur(BaseAttack):
    def __init__(self, radius=1.0):
        self.radius = radius

    def apply(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))

    def get_param_str(self):
        return f"Blur_{self.radius}"

@ATTACKS.register("GaussianNoise")
class GaussianNoise(BaseAttack):
    def __init__(self, std=0.1):
        self.std = std

    def apply(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image, dtype=np.uint8)
        
        g_noise = np.random.randn(*img_np.shape).astype(np.float32) * (self.std * 255) 
        
        noisy_array = np.clip(img_np.astype(np.float32) + g_noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    def get_param_str(self):
        return f"Noise_{self.std}"

@ATTACKS.register("CropRescale")
class CropRescale(BaseAttack):
    def __init__(self, scale=0.8):
        self.scale = scale

    def apply(self, image: Image.Image) -> Image.Image:
        import random
        W, H = image.size
        cW, cH = max(1, int(W * self.scale)), max(1, int(H * self.scale))
        left = random.randint(0, max(W - cW, 1))
        top = random.randint(0, max(H - cH, 1))
        cropped = image.crop((left, top, left + cW, top + cH))
        return cropped.resize((W, H), Image.BICUBIC)

    def get_param_str(self):
        return f"CropRes_{self.scale}"

@ATTACKS.register("RandomDrop")
class RandomDrop(BaseAttack):
    def __init__(self, ratio=0.8, seed=0):
        self.ratio = ratio
        self.seed = seed

    def apply(self, image: Image.Image) -> Image.Image:
        set_random_seed(self.seed)
        img_np = np.array(image)
        height, width, _ = img_np.shape
        
        crop_width = int(width * self.ratio)
        crop_height = int(height * self.ratio)
        
        start_x = np.random.randint(0, width - crop_width + 1)
        start_y = np.random.randint(0, height - crop_height + 1)
        
        padded_image = np.zeros_like(img_np)
        padded_image[start_y:start_y+crop_height, start_x:start_x+crop_width] =            img_np[start_y:start_y+crop_height, start_x:start_x+crop_width]
            
        return Image.fromarray(padded_image)

    def get_param_str(self):
        return f"Drop_{self.ratio}"

@ATTACKS.register("Rotation")
class RotationAttack(BaseAttack):
    def __init__(self, angle=25):
        self.angle = angle

    def apply(self, image: Image.Image) -> Image.Image:
        return image.rotate(self.angle, expand=False)

    def get_param_str(self):
        return f"Rot_{self.angle}"
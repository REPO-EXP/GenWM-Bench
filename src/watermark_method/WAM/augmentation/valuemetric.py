
import io
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from ..utils.image import jpeg_compress, median_filter
from ..data.transforms import normalize_img, unnormalize_img

class JPEG(nn.Module):
    def __init__(self, min_quality=None, max_quality=None, passthrough=True):
        super(JPEG, self).__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.passthrough = passthrough

    def jpeg_single(self, image, quality):
        if self.passthrough:
            return (jpeg_compress(image, quality).to(image.device) - image).detach() + image
        else:
            return jpeg_compress(image, quality).to(image.device)

    def forward(self, image: torch.tensor, mask, quality=None):
        if quality is None:
            if self.min_quality is None or self.max_quality is None:
                raise ValueError("Quality range must be specified")
            quality = torch.randint(self.min_quality, self.max_quality + 1, size=(1, )).item()
        image = unnormalize_img(image).clamp(0, 1)
        if len(image.shape) == 4:  
            for ii in range(image.shape[0]):
                image[ii] = self.jpeg_single(image[ii], quality)
        else:
            image = self.jpeg_single(image, quality)
        return normalize_img(image), mask

class GaussianBlur(nn.Module):
    def __init__(self, min_kernel_size=None, max_kernel_size=None):
        super(GaussianBlur, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size

    def forward(self, image, mask, kernel_size=None):
        if kernel_size is None:
            if self.min_kernel_size is None or self.max_kernel_size is None:
                raise ValueError("Kernel size range must be specified")
            kernel_size = torch.randint(self.min_kernel_size, self.max_kernel_size + 1, size=(1, )).item()
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        image = unnormalize_img(image).clamp(0, 1)
        image = F.gaussian_blur(image, kernel_size)
        return normalize_img(image), mask

class MedianFilter(nn.Module):
    def __init__(self, min_kernel_size=None, max_kernel_size=None, passthrough=True):
        super(MedianFilter, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.passthrough = passthrough

    def forward(self, image, mask, kernel_size=None):
        if kernel_size is None:
            if self.min_kernel_size is None or self.max_kernel_size is None:
                raise ValueError("Kernel size range must be specified")
            kernel_size = torch.randint(self.min_kernel_size, self.max_kernel_size + 1, size=(1, )).item()
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        image = unnormalize_img(image).clamp(0, 1)
        if self.passthrough:
            image = (median_filter(image, kernel_size) - image).detach() + image
        else:
            image = median_filter(image, kernel_size)
        return normalize_img(image), mask

class Brightness(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Brightness, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, image, mask, factor=None):
        if factor is None:
            if self.min_factor is None or self.max_factor is None:
                raise ValueError("min_factor and max_factor must be provided")
            factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        image = unnormalize_img(image)
        image = image.clamp(0, 1)
        image = F.adjust_brightness(image, factor)
        return normalize_img(image), mask

class Contrast(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Contrast, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, image, mask, factor=None):
        if factor is None:
            if self.min_factor is None or self.max_factor is None:
                raise ValueError("min_factor and max_factor must be provided")
            factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        image = unnormalize_img(image).clamp(0, 1)
        image = F.adjust_contrast(image, factor)
        return normalize_img(image), mask

class Saturation(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Saturation, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, image, mask, factor=None):
        if factor is None:
            if self.min_factor is None or self.max_factor is None:
                raise ValueError("Factor range must be specified")
            factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        image = unnormalize_img(image).clamp(0, 1)
        image = F.adjust_saturation(image, factor)
        return normalize_img(image), mask

class Hue(nn.Module):
    def __init__(self, min_factor=None, max_factor=None):
        super(Hue, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, image, mask, factor=None):
        if factor is None:
            if self.min_factor is None or self.max_factor is None:
                raise ValueError("Factor range must be specified")
            factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
        image = unnormalize_img(image).clamp(0, 1)
        image = F.adjust_hue(image, factor)
        return normalize_img(image), mask

if __name__ == "__main__":
    import os
    import torch
    from torchvision.transforms import ToTensor
    from torchvision.utils import save_image
    from PIL import Image

    from ..data.transforms import default_transform, unnormalize_img

    transformations = [
        (Brightness, [0.5, 1.5]),
        (Contrast, [0.5, 1.5]),
        (Saturation, [0.5, 1.5]),
        (Hue, [-0.5, -0.25, 0.25, 0.5]),
        (JPEG, [40, 60, 80]),
        (GaussianBlur, [3, 5, 9, 17]),
        (MedianFilter, [3, 5, 9, 17]),
        
    ]

    imgs = [
        Image.open("assets/images/gauguin_256.png")
    ]
    imgs = torch.stack([default_transform(img) for img in imgs])

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    for transform, strengths in transformations:
        for strength in strengths:
            
            transform_instance = transform()

            imgs_transformed, _ = transform_instance(imgs, None, strength)

            filename = f"{transform.__name__}_strength_{strength}.png"
            save_image(unnormalize_img(imgs_transformed).clamp(0, 1), os.path.join(output_dir, filename))

            print(f"Saved transformed images ({transform.__name__}, strength={strength}) to:", os.path.join(output_dir, filename))


import io
from PIL import Image

import torch
import torchvision.transforms as transforms

from ..data.metrics import bit_accuracy

def jpeg_compress(image: torch.Tensor, quality: int) -> torch.Tensor:
    
    assert image.min() >= 0 and image.max() <= 1, f'Image pixel values must be in the range [0, 1], got [{image.min()}, {image.max()}]'
    pil_image = transforms.ToPILImage()(image)  
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    
    buffer.seek(0)  
    compressed_image = Image.open(buffer)
    tensor_image = transforms.ToTensor()(compressed_image)
    return tensor_image

def webp_compress(image: torch.Tensor, quality: int) -> torch.Tensor:
    
    image = torch.clamp(image, 0, 1)  
    pil_image = transforms.ToPILImage()(image)  
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format='WebP', quality=quality)
    
    buffer.seek(0)  
    compressed_image = Image.open(buffer)
    tensor_image = transforms.ToTensor()(compressed_image)
    return tensor_image

def median_filter(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    
    padding = kernel_size // 2
    
    images_padded = torch.nn.functional.pad(images, (padding, padding, padding, padding))
    
    blocks = images_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)  
    
    medians = blocks.median(dim=-1).values.median(dim=-1).values  
    return medians

def create_diff_img(img1, img2):
    
    diff = img1 - img2
    
    diff = (diff - diff.min()) / ( (diff.max() - diff.min()) + 1e-6)
    return torch.abs(diff - 0.5)

def detect_wm_hm(preds, msgs, bit_accuracy_, params):
    
    mask_preds_hm = preds[:, 1:, :, :]  
    B, K, H, W = mask_preds_hm.shape
    msgs_expanded = msgs.view(B, K, 1, 1).expand(B, K, H, W)
    
    bit_matches = ((mask_preds_hm>0).float() == msgs_expanded).float()  
    bit_matches = bit_matches.mean(1, keepdim=True)
    
    mask_preds_hm = 100 * (bit_matches - params.threshold_mask)  
    dynamic_threshold = max(0.5, (bit_accuracy_+0.5)/2)
    mask_preds_hm_dynamic = 2 * (bit_matches - dynamic_threshold)  
    return mask_preds_hm, mask_preds_hm_dynamic

def masks_to_colored_image(masks, color_palette):
    batch_size, num_masks, H, W = masks.shape
       
    colored_images = torch.zeros((batch_size, 3, H, W), dtype=torch.uint8).to(masks.device)
       
    for i in range(num_masks):
        color = torch.tensor(color_palette[i], dtype=torch.uint8).view(3, 1, 1).to(masks.device)
        mask = masks[:, i, :, :].unsqueeze(1).to(torch.uint8)   
        colored_images += mask * color
       
    return colored_images

def create_fixed_color_palette(max_masks):
    
    colors = [
        (255, 255, 255), 
        (255, 0, 0),    
        (0, 255, 0),    
        (0, 0, 255),    
        (255, 255, 0),  
        (0, 255, 255),  
        (255, 0, 255),  
        (128, 0, 0),    
        (0, 128, 0),    
        (0, 0, 128),    
        
    ]
    
    if len(colors) < max_masks:
        
        colors.extend([tuple(np.random.choice(range(256), size=3)) for _ in range(max_masks - len(colors))])
    return colors[:max_masks]

if __name__ == '__main__':
    
    x = torch.rand(3, 256, 256)  
    x_jpeg = jpeg_compress(x, 80)  
    x_webp = webp_compress(x, 80)  

    print(x[0, 0:5, 0:5])  
    print(x_jpeg[0, 0:5, 0:5])  
    print(x_webp[0, 0:5, 0:5])  

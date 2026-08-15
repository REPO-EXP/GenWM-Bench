
import torch
from torch import nn

from ..modules.vit import ImageEncoderViT
from ..modules.pixel_decoder import PixelDecoder

class Extractor(nn.Module):
    
    def __init__(self) -> None:
        super(Extractor, self).__init__()

    def forward(
        self, 
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        
        return ...

class SegmentationExtractor(Extractor):
    
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        pixel_decoder: PixelDecoder,
    ) -> None:
        super(SegmentationExtractor, self).__init__()
        self.image_encoder = image_encoder
        self.pixel_decoder = pixel_decoder

    def forward(
        self, 
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        
        latents = self.image_encoder(imgs)
        masks = self.pixel_decoder(latents)

        return masks

def build_extractor(name, cfg, img_size, nbits):
    if name.startswith('sam'):
        cfg.encoder.img_size = img_size  
        cfg.pixel_decoder.nbits = nbits
        image_encoder = ImageEncoderViT(**cfg.encoder)
        pixel_decoder = PixelDecoder(**cfg.pixel_decoder)
        extractor = SegmentationExtractor(image_encoder=image_encoder, pixel_decoder=pixel_decoder)
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    return extractor
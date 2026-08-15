
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import Upsample

class PixelDecoder(nn.Module):
  def __init__(
    self,
    *,
    embed_dim: int,
    nbits: int = 0,
    activation: Type[nn.Module] = nn.GELU,
    upscale_stages: List[int] = [4, 2, 2],
    upscale_type: str = 'bilinear',
    sigmoid_output: bool = False,
  ) -> None:
    
    super().__init__()
    self.embed_dim = embed_dim
    self.nbits = nbits

    self.output_upscaling = []
    for up_factor in upscale_stages:
        self.output_upscaling += [
            Upsample(upscale_type, embed_dim, embed_dim // up_factor, up_factor, activation),
        ]
        embed_dim //= up_factor
    self.output_upscaling = nn.Sequential(*self.output_upscaling)

    self.out_channels = self.nbits + 1
    self.last_layer = nn.Conv2d(embed_dim, self.out_channels, kernel_size=1, bias=True)
    self.sigmoid_output = sigmoid_output

  def forward(
    self,
    image_embeddings: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    
    b, c, h, w = image_embeddings.shape  

    upscaled_embedding = self.output_upscaling(image_embeddings)  
    preds = self.last_layer(upscaled_embedding)  
    if self.sigmoid_output: 
      preds = F.sigmoid(preds)

    return preds

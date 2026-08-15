
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, apply_forward_hook
from .modeling_utils import ModelMixin
from .vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder

@dataclass
class AutoencoderKLOutput(BaseOutput):
    
    latent_dist: "DiagonalGaussianDistribution"

class AutoencoderKL(ModelMixin, ConfigMixin):
    
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.use_slicing = False

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, s: torch.FloatTensor, able, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        s = self.post_quant_conv(s)
        dec = self.decoder(z, s, able)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def enable_slicing(self):
        
        self.use_slicing = True

    def disable_slicing(self):
        
        self.use_slicing = False

    @apply_forward_hook
    def decode(self, z: torch.FloatTensor, s: torch.FloatTensor, able, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice, s, able).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, s, able).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

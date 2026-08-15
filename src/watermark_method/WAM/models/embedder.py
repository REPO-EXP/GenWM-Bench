
import torch
from torch import nn

from ..modules.vae import VAEEncoder, VAEDecoder
from ..modules.msg_processor import MsgProcessor

class Embedder(nn.Module):
    
    def __init__(self) -> None:
        super(Embedder, self).__init__()
    
    def get_random_msg(self, bsz: int = 1, nb_repetitions = 1) -> torch.Tensor:
        
        return ...

    def get_last_layer(self) -> torch.Tensor:
        return None

    def forward(
        self, 
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        
        return ...

class VAEEmbedder(Embedder):
    
    def __init__(
        self,
        encoder: VAEEncoder,
        decoder: VAEDecoder,
        msg_processor: MsgProcessor
    ) -> None:
        super(VAEEmbedder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.msg_processor = msg_processor

    def get_random_msg(self, bsz: int = 1, nb_repetitions = 1) -> torch.Tensor:
        return self.msg_processor.get_random_msg(bsz, nb_repetitions)  

    def get_last_layer(self) -> torch.Tensor:
        last_layer = self.decoder.conv_out.weight
        return last_layer

    def forward(
        self, 
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        
        latents = self.encoder(imgs)
        latents_w = self.msg_processor(latents, msgs)
        imgs_w = self.decoder(latents_w)
        return imgs_w

def build_embedder(name, cfg, nbits):
    if name.startswith('vae'):
        
        cfg.msg_processor.nbits = nbits
        cfg.msg_processor.hidden_size = nbits * 2
        cfg.decoder.z_channels = (nbits * 2) + cfg.encoder.z_channels
        
        encoder = VAEEncoder(**cfg.encoder)
        msg_processor = MsgProcessor(**cfg.msg_processor)
        decoder = VAEDecoder(**cfg.decoder)
        embedder = VAEEmbedder(encoder, decoder, msg_processor)
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    return embedder

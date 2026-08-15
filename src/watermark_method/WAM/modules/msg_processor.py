
import torch

class MsgProcessor(torch.nn.Module):
    
    def __init__(
        self, 
        nbits: int, 
        hidden_size: int,
        msg_processor_type: str = "binary+concat",
        msg_mult: float = 1.0,
    ):
        super().__init__()
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_mult = msg_mult
        
        self.msg_processor_type = msg_processor_type if nbits > 0 else "none+_"
        self.msg_type = self.msg_processor_type.split("+")[0]
        self.msg_agg = self.msg_processor_type.split("+")[1]
        
        if self.msg_type.startswith("no"):  
            self.msg_embeddings = torch.tensor([])
        elif self.msg_type.startswith("bin"):  
            self.msg_embeddings = torch.nn.Embedding(2 * nbits, hidden_size)
        elif self.msg_type.startswith("gau"):  
            self.msg_embeddings = torch.nn.Embedding(nbits, hidden_size)
        else:
            raise ValueError(f"Invalid msg_processor_type: {self.msg_processor_type}")

    def get_random_msg(self, bsz: int = 1, nb_repetitions = 1) -> torch.Tensor:
        
        if self.msg_type.startswith("bin"):
            if nb_repetitions != 1:
                assert self.nbits % nb_repetitions == 0, f"nbits must be divisible by nb_repetitions, got {self.nbits} and {nb_repetitions}"
                aux = torch.randint(0, 2, (bsz, self.nbits // nb_repetitions)) 
                return aux.unsqueeze(1).repeat(1, nb_repetitions, 1).view(bsz, self.nbits)
            else:
                return torch.randint(0, 2, (bsz, self.nbits))
        elif self.msg_type.startswith("gau"):
            gauss_vecs = torch.randn(bsz, self.nbits)
            gauss_vecs = gauss_vecs / torch.norm(gauss_vecs, dim=-1, keepdim=True)
            return gauss_vecs
        return torch.tensor([])

    def forward(
        self, 
        latents: torch.Tensor, 
        msg: torch.Tensor,
        verbose: bool = False
    ) -> torch.Tensor:
        
        if self.nbits == 0:
            return latents

        if self.msg_type.startswith("bin"):
            
            indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)  
            indices = indices.repeat(msg.shape[0], 1)  
            indices = (indices + msg).long()
            
            msg_aux = self.msg_embeddings(indices)  
            msg_aux = msg_aux.sum(dim=-2)  
            msg_aux = msg_aux.unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, latents.shape[-2], latents.shape[-1]
            )  
        elif self.msg_type.startswith("gau"):
            
            indices = torch.arange(msg.shape[-1]).to(msg.device).long()  
            msg_aux = self.msg_embeddings(indices)  
            msg_aux = torch.einsum("kd, bk -> bd", msg_aux, msg)  
            msg_aux = msg_aux.unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, latents.shape[-2], latents.shape[-1]
            )  
        else:
            raise ValueError(f"Invalid msg_type: {self.msg_type}")

        if self.msg_agg == "concat":
            latents = torch.cat([
                latents,  
                self.msg_mult * msg_aux  
            ], dim=1)  
        elif self.msg_agg == "add":
            latents = latents + self.msg_mult * msg_aux  
        else:
            raise ValueError(f"Invalid msg_agg: {self.msg_agg}")
        
        if verbose:    
            print(f'indices: {indices.shape}')
            print(f'msgs: {msg.shape}')
            print(f'msg_aux: {msg_aux.shape}')
            print(f'latents: {latents.shape}')

        return latents

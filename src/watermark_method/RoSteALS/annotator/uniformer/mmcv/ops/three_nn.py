from typing import Tuple

import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['three_nn_forward'])

class ThreeNN(Function):
    
    @staticmethod
    def forward(ctx, target: torch.Tensor,
                source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        target = target.contiguous()
        source = source.contiguous()

        B, N, _ = target.size()
        m = source.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        ext_module.three_nn_forward(target, source, dist2, idx, b=B, n=N, m=m)
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(idx)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

three_nn = ThreeNN.apply

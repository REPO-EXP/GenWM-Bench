
import torch
import torch.nn as nn
from einops import rearrange

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context); v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))
    def forward(self, x):
        return self.net(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, disable_self_attn=False, checkpoint=True):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.checkpoint = checkpoint
        inner_dim = n_heads * d_head
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, context_dim=None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    def forward(self, x, context=None):
        if not self.disable_self_attn:
            x = self.attn1(self.norm1(x)) + x
        if context is not None:
            x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None, use_checkpoint=True, disable_self_attn=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                  disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
            for _ in range(depth)
        ])
        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in

class MemoryEfficientCrossAttention(CrossAttention):
    pass

def default(val, d):
    return val if val is not None else d

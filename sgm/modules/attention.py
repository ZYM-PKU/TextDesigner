import math
from inspect import isfunction
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from torchvision.utils import save_image

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("No module 'xformers'.")


def exists(val):
    return val is not None

def uniq(arr):
    return {el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )

def AdaIN(feat, ref, mask):

    b, n, d = feat.shape

    ref_std = ref.std(dim=-2, keepdims=True) # (b, 1, d)
    ref_mean = ref.mean(dim=-2, keepdims=True) # (b, 1, d)

    if mask is None:
        mask = torch.ones((b, n, 1)).to(device=feat.device)

    masked_feats = [torch.masked_select(f, m.bool()).reshape((-1, d)) for f, m in zip(feat, mask)] # (b, x, d)
    
    feat_std = torch.cat([mf.std(dim=-2, keepdims=True) for mf in masked_feats]).unsqueeze(1) # (b, 1, d)
    feat_mean = torch.cat([mf.mean(dim=-2, keepdims=True) for mf in masked_feats]).unsqueeze(1) # (b, 1, d)

    new_feat = (feat - feat_mean) / feat_std
    new_feat = new_feat * ref_std + ref_mean

    feat = new_feat * mask + feat * (1 - mask)

    return feat


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SelfAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        heads=8,
        dim_head=64,
        dropout=0.0
    ):
        super().__init__()

        inner_dim = dim_head * heads

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = zero_module(
            nn.Sequential(
                nn.Linear(inner_dim, query_dim),
                nn.Dropout(dropout)
            )
        )

    def forward(
        self,
        x,
        mask,
        share=False
    ):
        
        h = self.heads

        q = self.to_q(x) # (2b, n, d)
        k = self.to_k(x) # (2b, n, d)
        v = self.to_v(x) # (2b, n, d)

        if share:

            n = q.shape[1]
            size = int(n**0.5)
            if mask is not None:
                mask = F.interpolate(mask, (size, size)).reshape((-1, n, 1)) # (b, n, 1)

            ref_k, feat_k = k.chunk(2) # ref_k (b, n, d) feat_k (b, n, d)
            ref_v, feat_v = v.chunk(2) # ref_v (b, n, d) feat_v (b, n, d)

            feat_k = AdaIN(feat_k, ref_k, mask) # feat_k (b, n, d)
            feat_v = AdaIN(feat_v, ref_v, mask) # feat_v (b, n, d)

            k = torch.cat([ref_k, feat_k], dim=0) # (2b, n, d)
            v = torch.cat([ref_v, feat_v], dim=0) # (2b, n, d)

            k = torch.cat([k, ref_k.tile((2, 1, 1))], dim=1) # (2b, 2n, d)
            v = torch.cat([v, ref_v.tile((2, 1, 1))], dim=1) # (2b, 2n, d)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale # (2bh, n, 2n)
        del q, k

        if share and mask is not None:

            mask = mask.tile((h, 1, n)) # (bh, n, n)
            mask = (1 - mask) * (-1e9)
            mask = torch.cat([torch.zeros_like(mask), mask], dim=0) # (2bh, n, n)
            mask = torch.cat([torch.zeros_like(mask), mask], dim=-1) # (2bh, n, 2*n)

            sim = sim + mask

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1) # softmax on token dim

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h) # (2b, n, d)
        
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0
    ):
        super().__init__()

        inner_dim = dim_head * heads

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = zero_module(
            nn.Sequential(
                nn.Linear(inner_dim, query_dim),
                nn.Dropout(dropout)
            )
        )

        self.attn_map_cache = None
        self.min_attn_size = 32

    def forward(
        self,
        x,
        context=None,
        mask=None
    ):
        
        h = self.heads
        b, n = x.shape[:2]
        l = context.shape[1]
        size = int(n**0.5)

        q = self.to_q(x) # (b, n, hd)
        k = self.to_k(context) # (b, l, hd)
        v = self.to_v(context) # (b, l, hd)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)) # (bh, -1, d)

        # attention, what we cannot get enough of
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale # (bh, n, l)
        del q, k

        # apply mask
        if mask is not None and size >= self.min_attn_size:
            bias = torch.zeros_like(sim) # (bh, n, l)
            mask = F.interpolate(mask, size=size) # (b, l, s, s)
            mask = mask.reshape(b, l, -1).permute(0, 2, 1).contiguous() # (b, n, l)
            mask = mask.tile(h, 1, 1).to(dtype=torch.bool) # (bh, n, l)
            bias.masked_fill_(mask.logical_not(), -1.0e+9) # (bh, n, l)
            sim += bias

        sim = sim.softmax(dim=-1) # softmax on token dim

        # save attn_map
        if self.attn_map_cache is not None:
            self.attn_map_cache["size"] = size
            self.attn_map_cache["attn_map"] = sim

        out = einsum('b i j, b j d -> b i d', sim, v) # (bh, n, d)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h) # (b, n, hd)
        
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        # print(
        #     f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
        #     f"{heads} heads with a dimension of {dim_head}."
        # )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        t_context_dim=None,
        v_context_dim=None,
        gated_ff=True
    ):
        super().__init__()

        # self-attention
        self.attn1 = MemoryEfficientCrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout
        )

        # textual cross-attention
        if t_context_dim is not None and t_context_dim > 0:
            self.t_attn = CrossAttention(
                query_dim=dim,
                context_dim=t_context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout
            )
            self.t_norm = nn.LayerNorm(dim)
        
        # visual cross-attention
        if v_context_dim is not None and v_context_dim > 0:
            self.v_attn = CrossAttention(
                query_dim=dim,
                context_dim=v_context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout
            )
            self.v_norm = nn.LayerNorm(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

    def forward(self, x, t_context=None, v_context=None, mask=None):
        x = (
            self.attn1(
                self.norm1(x),
            )
            + x
        )
        if hasattr(self, "t_attn"):
            x = (
                self.t_attn(
                    self.t_norm(x),
                    context=t_context,
                    mask=mask
                )
                + x
            )
        if hasattr(self, "v_attn"):
            x = (
                self.v_attn(
                    self.v_norm(x),
                    context=v_context,
                    mask=mask
                )
                + x
            )

        x = self.ff(self.norm3(x)) + x

        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        t_context_dim=None,
        v_context_dim=None,
        use_linear=False
    ):
        super().__init__()
 
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    t_context_dim=t_context_dim,
                    v_context_dim=v_context_dim
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, t_context=None, v_context=None, mask=None):

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, t_context=t_context, v_context=v_context, mask=mask)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)

        return x + x_in
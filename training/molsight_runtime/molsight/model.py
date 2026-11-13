from contextlib import contextmanager, nullcontext
import os
from typing import Dict, Iterable, List, Optional, Tuple, Union, Callable
import numpy as np

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import timm

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

from .decoding import TokenDecoder, GreedyDecoder, MaximumLikelihoodRanker
from .tokenizer import SOS_ID, EOS_ID, PAD_ID
from .vary_b import build_vary_vit_b
from .position import resize_fourier_grid


class EdgePredictor(nn.Module):

    def __init__(self, decoder_dim):
        super(EdgePredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, 7)
        )

    #@torch.compile
    def _forward(self, hidden):
        b, l, dim = hidden.size()
        hh = torch.cat([hidden.unsqueeze(2).expand(b, l, l, dim), hidden.unsqueeze(1).expand(b, l, l, dim)], dim=3)
        results = self.mlp(hh)  # [b, l, l, 7]

        return results
    
    def forward(self, hidden, indices):
        b, l, dim = hidden.size()
        batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
        indices = indices.view(-1) - 1  # Output of the previous token represent the next token, so we need to subtract 1 
        hidden = hidden[batch_id, indices].view(b, -1, dim)

        return self._forward(hidden)   # [b, l, l, 7]


class LocationPredictor(nn.Module):

    def __init__(self, decoder_dim, n_coord_bins, init_scale=0.1):
        super(LocationPredictor, self).__init__()
        '''self.coords_mlp = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, 2)
        )'''

        pe = torch.zeros(n_coord_bins, decoder_dim)
        theta = 10000
        # Compute the inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, decoder_dim, 2, dtype=torch.int64).to(dtype=torch.float) / decoder_dim))
        inv_freq_expanded = inv_freq[:, None].float()  # [dim/2, 1]
        position_ids = torch.arange(n_coord_bins)
        position_ids_expanded = position_ids[None, :].float()    # [1, max_len]

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).T # [max_len, dim/2]

        # sin for even indices, cos for odd indices
        pe[:, 0::2] = torch.sin(freqs)  # even indices
        pe[:, 1::2] = torch.cos(freqs)  # odd indices

        self.register_buffer('pe', pe, persistent=False)

        self.x_fc = nn.Linear(decoder_dim, decoder_dim)
        self.y_fc = nn.Linear(decoder_dim, decoder_dim)
        self.emb_mlp = nn.Linear(decoder_dim, decoder_dim)

        self.register_buffer('x_bins', torch.linspace(0, 1, n_coord_bins), persistent=False)
        self.register_buffer('y_bins', torch.linspace(0, 1, n_coord_bins), persistent=False)

        # self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float), requires_grad=True)  # a learnable scale factor
        # fully-connected layers to predict sigma
        self.sigma_fc = nn.Linear(decoder_dim, 1)
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of the model."""
        nn.init.constant_(self.sigma_fc.weight, 0.0)
        nn.init.constant_(self.sigma_fc.bias, 0.0)

    #@torch.compile
    def _forward(self, hidden):
        #results = self.coords_mlp(hidden)   # [b, l, 2]
        kpt_feats = self.emb_mlp(hidden)  # [b, l, decoder_dim]
        sigmas = F.softplus(self.sigma_fc(hidden)) + 1e-3  # [b, l, 1]
        x_bins_enc = self.x_fc(self.pe)
        y_bins_enc = self.y_fc(self.pe)    # [n_coord_bins, decoder_dim]

        x_hms = torch.matmul(kpt_feats,
                             x_bins_enc.transpose(-1, -2).contiguous())
        y_hms = torch.matmul(kpt_feats,
                             y_bins_enc.transpose(-1, -2).contiguous())    # [b, l, n_coord_bins]
        x_hms = x_hms.softmax(dim=-1)
        y_hms = y_hms.softmax(dim=-1)   # [b, l, n_coord_bins]

        # computes the weighted sum(expectation) of these bins to derive the x and y coordinates.
        x = torch.matmul(x_hms, self.x_bins.unsqueeze(1))
        y = torch.matmul(y_hms, self.y_bins.unsqueeze(1))   # [b, l, 1]
        loc_pred = torch.cat([x, y], dim=-1)  # [b, l, 2]

        return loc_pred, x_hms, y_hms, sigmas

    def forward(self, hidden, indices):
        b, l, dim = hidden.size()
        batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
        indices = indices.view(-1) - 1
        hidden = hidden[batch_id, indices].view(b, -1, dim)
        
        return self._forward(hidden)  # [b, l, 2], [b, l, n_coord_bins], [b, l, n_coord_bins], [b, l, 1]
    
class LocationPredictorRegression(nn.Module):

    def __init__(self, decoder_dim):
        super(LocationPredictorRegression, self).__init__()
        self.coords_mlp = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, 2)
        )

    #@torch.compile
    def _forward(self, hidden):
        loc_pred = self.coords_mlp(hidden)   # [b, l, 2]
        loc_pred = F.sigmoid(loc_pred)

        return loc_pred, None, None, None

    def forward(self, hidden, indices):
        b, l, dim = hidden.size()
        batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
        indices = indices.view(-1) - 1
        hidden = hidden[batch_id, indices].view(b, -1, dim)
        
        return self._forward(hidden)  # [b, l, 2], None, None, None


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    #@torch.compile
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
    

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int=512, theta=10000,):
        super().__init__()
        self.dim = dim
        # Compute the inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, dim).
            position_ids (torch.Tensor): Position IDs of shape (B, L).
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)  # [B, dim/2, 1]
        position_ids_expanded = position_ids[:, None, :].float()    # [B, 1, L]

        with torch.cuda.amp.autocast(enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # [B, L, dim/2]
            emb = torch.cat((freqs, freqs), dim=-1) # [B, L, dim]
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor. [B, n_head, 1 or L, dim/n_head]
        k (`torch.Tensor`): The key tensor. [B, n_head, L, dim/n_head]
        cos (`torch.Tensor`): The cosine part of the rotary embedding. [1, L, dim/n_head]
        sin (`torch.Tensor`): The sine part of the rotary embedding. [1, L, dim/n_head]
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.shape[2]    # 1 or L
    assert q_len in [1, k.shape[2]], f"q_len {q_len} must be 1 or equal to k.shape[2] {k.shape[2]}"
    q_embed = (q * cos[:, :, -q_len:]) + (rotate_half(q) * sin[:, :, -q_len:])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def enable_lora(model):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enable_adapter()

def disable_lora(model):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.disable_adapter()

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear
        in_dim, out_dim = base_linear.in_features, base_linear.out_features
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)

        self.lora_dropout = nn.Dropout(dropout)
        self.enabled = True

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base(x)
        if self.enabled:
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return base_out + lora_out
        else:
            return base_out

    def enable_adapter(self):
        self.enabled = True

    def disable_adapter(self):
        self.enabled = False

def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int, lora: bool = False):
        super().__init__()
        self.n_head = n_head
        if lora:
            self.query = LoRALinear(Linear(n_state, n_state), r=8, alpha=16)
            self.key = LoRALinear(Linear(n_state, n_state, bias=False), r=8, alpha=16)
            self.value = LoRALinear(Linear(n_state, n_state), r=8, alpha=16)
            self.out = LoRALinear(Linear(n_state, n_state), r=8, alpha=16)
        else:
            self.query = Linear(n_state, n_state)
            self.key = Linear(n_state, n_state, bias=False)
            self.value = Linear(n_state, n_state)
            self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xi: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        position_embeddings: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xi is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xi is None else xi)
            v = self.value(x if xi is None else xi)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask, position_embeddings)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, position_embeddings: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, C = q.shape
        scale = (C // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)   # [B, n_head, L, dim/n_head]
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if position_embeddings is not None:
            # rope
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and L > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:L, :L]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk


class SMILESDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_swiglu=False, 
        use_rmsnorm=False,
        lora: bool = False,
        **block_kwargs
    ):
        super().__init__()
        
        if not use_rmsnorm:
            self.self_attn_norm = LayerNorm(hidden_size)
            self.cross_attn_ln = LayerNorm(hidden_size)
            self.mlp_ln = LayerNorm(hidden_size)
        else:
            self.self_attn_norm = RMSNorm(hidden_size)
            self.cross_attn_ln = RMSNorm(hidden_size)
            self.mlp_ln = RMSNorm(hidden_size)

        self.attn = MultiHeadAttention(hidden_size, num_heads, lora)
        self.cross_attn = (
            MultiHeadAttention(hidden_size, num_heads, lora)
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = nn.Sequential(
                Linear(hidden_size, mlp_hidden_dim), nn.GELU(approximate="tanh"), Linear(mlp_hidden_dim, hidden_size)
            )
        

    def forward(
        self,
        x: Tensor,
        xi: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.self_attn_norm(x), mask=mask, position_embeddings=position_embeddings, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xi, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

def checkpoint_context(module):
    modules = list(module.modules())
    training = [child.training for child in modules]
    adapters = [(child, child.enabled) for child in modules if isinstance(child, LoRALinear)]

    @contextmanager
    def recompute():
        current_training = [child.training for child in modules]
        current_adapters = [child.enabled for child, _ in adapters]
        try:
            for child, state in zip(modules, training):
                child.training = state
            for child, enabled in adapters:
                child.enabled = enabled
            yield
        finally:
            for child, state in zip(modules, current_training):
                child.training = state
            for (child, _), enabled in zip(adapters, current_adapters):
                child.enabled = enabled

    return nullcontext(), recompute()


class SMILESDecoder(nn.Module):
    def __init__(
        self, 
        args,
        n_vocab: int, 
    ):
        super().__init__()

        max_len = args.max_len
        hidden_size = args.embed_dim
        n_head = args.dec_n_head
        n_layer = args.dec_n_layer
        
        self.use_checkpoint  = args.use_checkpoint
        self.use_qknorm = args.use_qknorm
        
        self.token_embedding = nn.Embedding(n_vocab, hidden_size, padding_idx=PAD_ID)
        if self.use_qknorm:
            # self.rotary_emb = RotaryEmbedding(dim=hidden_size // n_head)
            head_dim = hidden_size // n_head
            theta = 10000
            # Compute the inverse frequencies
            inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
            inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1)  # [1, dim/2, 1]
            position_ids = torch.arange(max_len)
            position_ids_expanded = position_ids.expand(1, 1, -1).float()    # [1, 1, max_len]

            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # [1, max_len, dim/2]
            emb = torch.cat((freqs, freqs), dim=-1) # [1, max_len, dim]
            self.register_buffer("embed_cos", emb.cos(), persistent=False)
            self.register_buffer("embed_sin", emb.sin(), persistent=False)
        else:
            self.positional_embedding = nn.Embedding(max_len, hidden_size)

        self.blocks: Iterable[SMILESDecoderBlock] = nn.ModuleList(
            [
                SMILESDecoderBlock(hidden_size, n_head, use_swiglu=args.use_swiglu, use_rmsnorm=args.use_rmsnorm, lora=args.lora)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(hidden_size)

        mask = torch.empty(max_len, max_len).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

        self.edge_predictor = self.loc_predictor = None
        if 'edges' in args.formats:
            self.edge_predictor = EdgePredictor(hidden_size)
        if 'coords' in args.formats:
            self.loc_predictor = LocationPredictor(hidden_size, args.n_coord_bins) if not args.regression else LocationPredictorRegression(hidden_size)
            loc_blocks: Iterable[SMILESDecoderBlock] = nn.ModuleList(
                [
                    SMILESDecoderBlock(hidden_size, n_head, use_swiglu=args.use_swiglu, use_rmsnorm=args.use_rmsnorm)
                    for _ in range(2)
                ]
            )
            self.loc_predictor.loc_blocks = loc_blocks

    def forward(self, x: Tensor, xi: Tensor, atom_indices: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= max_len)
            the text tokens
        xi : torch.Tensor, shape = (batch_size, H x W, C)
            the encoded image features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0  # num of processed tokens
        x = self.token_embedding(x)    # [B, L, C]

        if self.use_qknorm:
            # create position embeddings to be shared across the decoder layers
            # position_ids = torch.arange(offset, offset + x.shape[1], device=x.device).unsqueeze(0)
            total_len = x.shape[1] + offset
            position_embeddings = (self.embed_cos[:, :total_len].to(xi.dtype), self.embed_sin[:, :total_len].to(xi.dtype))
        else:
            x = x + self.positional_embedding.weight[offset : offset + x.shape[1]]
            position_embeddings = None
        
        x = x.to(xi.dtype)

        for block in self.blocks:
            if self.use_checkpoint and torch.is_grad_enabled():
                x = checkpoint(
                    block, x, xi, self.mask, position_embeddings, kv_cache,
                    use_reentrant=False, context_fn=lambda block=block: checkpoint_context(block),
                )
            else:
                x = block(x, xi, mask=self.mask, position_embeddings=position_embeddings, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        edge_pred, loc_pred = None, None
        if atom_indices is not None:
            if self.edge_predictor is not None:
                edge_pred = self.edge_predictor(x, atom_indices)
            if self.loc_predictor is not None:
                for block in self.loc_predictor.loc_blocks:
                    x = block(x, xi, position_embeddings=position_embeddings)
                loc_pred = self.loc_predictor(x, atom_indices)

        return x, logits, edge_pred, loc_pred


class MolsightModel(nn.Module):
    def __init__(
        self, 
        args, 
        tokenizer, 
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.initial_tokens = [SOS_ID]
        self.logit_filters = []
        self.sample_begin: int = len(self.initial_tokens)

        if args.encoder == 'efficientvit':
            self.image_embedding_size = tuple(getattr(
                args, 'image_pos_grid_size', (args.input_size // 8, args.input_size // 8)
            ))
            self.register_buffer(
                "image_pos_embed",
                get_2d_rff_pos_embed(self.image_embedding_size, args.embed_dim // 2),
            )

            pretrained = not (args.resume or getattr(args, 'load_path', None) or getattr(args, 'no_pretrained', False))
            self.image_encoder = timm.create_model('efficientvit_l1', pretrained=pretrained)
            # remove head of the image encoder
            self.image_encoder.head = nn.Identity()
            self.scale_factors = [1, 2, 4]
            self.num_channels = [int(ratio * self.image_encoder.num_features) for ratio in [0.25, 0.5, 1]]
            self.image_encoder.fuse_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.Identity() if scale == 1 else nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
                )
                for dim, scale in zip(self.num_channels, self.scale_factors)
            ])
            self.image_encoder.point_conv = nn.Sequential(
                nn.Conv2d(sum(self.num_channels), args.embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(args.embed_dim),
                nn.GELU(),
            )
            self.get_image_features = self.efficientvit_forward
        elif args.encoder == 'vitdet':
            self.image_embedding_size = tuple(getattr(
                args, 'image_pos_grid_size', (args.input_size // 32, args.input_size // 32)
            ))
            self.register_buffer(
                "image_pos_embed",
                get_2d_rff_pos_embed(self.image_embedding_size, args.embed_dim // 2),
            )
            self.image_encoder = build_vary_vit_b()
            # use got-ocr 2.0 weight for initialization
            assert os.path.exists(args.vitdet_weight), f"Vary ViT weight file {args.vitdet_weight} does not exist"
            from safetensors.torch import load_file
            state_dict = load_file(args.vitdet_weight)
            state_dict = {
                k[len('model.vision_tower_high.'):]: v for k, v in state_dict.items() if k.startswith('model.vision_tower_high.')
            }
            self.image_encoder.load_state_dict(state_dict)
            self.image_encoder.net_3 = nn.Identity()
            self.get_image_features = self.vitdet_forward
        else:
            raise ValueError(f"Unknown image encoder {args.encoder}")

        self.decoder = SMILESDecoder(args, len(tokenizer))
        if getattr(args, "compile_layers", False):
            for module in self.decoder.modules():
                if isinstance(module, (LayerNorm, SwiGLUFFN)):
                    module.compile(dynamic=True)

        self.max_len = args.max_len
    
    def efficientvit_forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract image features from the input image tensor."""
        intermediates = self.image_encoder.forward_intermediates(image, indices=len(self.scale_factors), intermediates_only=True)
        feats = []
        for inter_feat, fuse_block in zip(intermediates, self.image_encoder.fuse_blocks):
            feats.append(fuse_block(inter_feat))
        # sum up the features
        image_features = torch.cat(feats, dim=1)  # [B, C, H, W]
        image_features = self.image_encoder.point_conv(image_features)

        return image_features + self.image_positions(image_features)
    
    def vitdet_forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract image features from the input image tensor."""
        image_features = self.image_encoder(image)  # [B, C, H, W]

        return image_features + self.image_positions(image_features)

    def image_positions(self, features):
        size = tuple(features.shape[-2:])
        if size == tuple(self.image_pos_embed.shape[-2:]):
            return self.image_pos_embed
        if self.image_pos_embed.is_inference():
            return resize_fourier_grid(self.image_pos_embed, size)
        key = (self.image_pos_embed.data_ptr(), self.image_pos_embed._version,
               self.image_pos_embed.device, self.image_pos_embed.dtype, size)
        cached = getattr(self, '_image_position_cache', None)
        if cached is None or cached[0] != key:
            cached = (key, resize_fourier_grid(self.image_pos_embed, size))
            self._image_position_cache = cached
        return cached[1]
    
    def forward(
        self, image: torch.Tensor, label: torch.Tensor, atom_indices: Optional[Tensor] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        if kwargs.get('image_features') is not None:
            # if image features are already provided, use them directly
            image_features: Tensor = kwargs['image_features']
        else:
            image_features: Tensor = self.get_image_features(image)
        repeats = kwargs.get('feature_repeats', 1)
        if type(repeats) is not int or repeats < 1:
            raise ValueError("Image feature repeat count must be a positive integer")
        if repeats != 1:
            image_features = image_features.repeat_interleave(repeats, dim=0)
        assert image_features.shape[0] == label.shape[0], f"Image features batch size {image_features.shape[0]} does not match label batch size {label.shape[0]}"

        hidden_states, logits, edge_pred, loc_pred = self.decoder(label, image_features.flatten(2).permute(0, 2, 1), atom_indices=atom_indices)

        return {"logits": logits, "edge_pred": edge_pred, "loc_pred": loc_pred}
    
    def generate(self, image, kv_cache, decode_strategy=None, n_samples=1, **kwargs) -> Dict[str, List]:
        """Inference mode"""
        if decode_strategy is None:
            decode_strategy = GreedyDecoder(temperature=0, eot=EOS_ID)
        # sequence ranker: implements how to rank a group of sampled sequences
        sequence_ranker = MaximumLikelihoodRanker(use_length_penalty=False) if n_samples <= 1 else None

        n_img: int = image.shape[0]

        if kwargs.get('image_features') is not None:
            # if image features are already provided, use them directly
            image_features: Tensor = kwargs['image_features']
            assert image_features.shape[0] == n_img, f"Image features batch size {image_features.shape[0]} does not match input image batch size {n_img}"
        else:
            image_features: Tensor = self.get_image_features(image)
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_img, 1)   # [n_img, 1]

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        image_features = image_features.repeat_interleave(n_samples, dim=0)  # [n_img * n_group, C, H, W]
        tokens = tokens.repeat_interleave(n_samples, dim=0).to(image_features.device)    # [n_img * n_group, 1]

        def _main_loop(image_features: Tensor, tokens: Tensor, kv_cache: dict, decode_strategy: TokenDecoder):
            all_hidden_states = []
            all_logits = []
            n_batch = tokens.shape[0]
            sum_logprobs: Tensor = torch.zeros(n_batch, device=image_features.device)

            for i in range(self.max_len):
                hidden_states, logits, edge_pred, loc_pred = self.decoder(tokens[:, -1:],  # [n_img * n_group, 1]
                                                        image_features.flatten(2).permute(0, 2, 1), 
                                                        kv_cache=kv_cache)
                all_hidden_states.append(hidden_states)    # [n_img * n_group, 1, hidden_size]

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]  # [n_img * n_group, vocab_size]
                all_logits.append(logits.unsqueeze(1))  # [n_img * n_group, 1, vocab_size]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = decode_strategy.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] >= self.max_len:
                    break

            hidden_states = torch.cat(all_hidden_states, dim=1)  # [n_img * n_group, seq_len, hidden_size]
            logits = torch.cat(all_logits, dim=1)
            return hidden_states, logits, tokens, sum_logprobs
        
        # call the main sampling loop
        hidden_states, logits, tokens, sum_logprobs = _main_loop(image_features, tokens, kv_cache, decode_strategy)
        raw_tokens = tokens.detach().clone()  # keep the raw tokens for later processing

        hidden_states = hidden_states.reshape(n_img, n_samples, -1, hidden_states.shape[-1])
        tokens = tokens.reshape(n_img, n_samples, -1)
        sum_logprobs = sum_logprobs.reshape(n_img, n_samples)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = decode_strategy.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == EOS_ID).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        if sequence_ranker is not None:
            # select the top-ranked sample in each group
            selected = sequence_ranker.rank(tokens, sum_logprobs)
            hidden_states = torch.stack([hidden_states[i, selected[i]] for i in range(n_img)], dim=0)  # [n_img, seq_len, hidden_size]
            tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
            sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        else:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])
            _tokens = []
            for s in tokens:
                for t in s:
                    _tokens.append(t.tolist())
            tokens = _tokens
            _sum_logprobs = []
            for lp in sum_logprobs:
                _sum_logprobs.extend(lp)
            sum_logprobs = _sum_logprobs

        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]
        
        batch_preds: List[Dict[str, List]] = [self.tokenizer.sequence_to_smiles(seq) for seq in tokens]
        batch_preds: Dict[str, List] = {key: [d[key] for d in batch_preds] for key in batch_preds[0].keys()}

        batch_preds['tokens'] = tokens
        batch_preds['avg_logprob'] = avg_logprobs

        return batch_preds, {
            'image_features': image_features.flatten(2).permute(0, 2, 1),
            'hidden_states': hidden_states,
            'raw_tokens': raw_tokens,
            'token_logps': F.log_softmax(logits.float() / (getattr(decode_strategy, 'temperature', 1.0) or 1.0), dim=-1).gather(
                -1, raw_tokens[:, 1:].unsqueeze(-1)
            ).squeeze(-1).detach(),
        }
    
    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > 1:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks
    
def get_2d_rff_pos_embed(spatial_size: Tuple[int, int], half_embed_dim: int) -> torch.Tensor:
    """
    Generate 2D random Fourier features positional embedding.
    Args:
        spatial_size (tuple): Size of the spatial dimension (height, width).
        half_embed_dim (int): Half of the embedding dimension.
    Returns:
        torch.Tensor: Positional embedding tensor of shape (1, embed_dim, height, width).
    """
    h, w = spatial_size
    grid = torch.ones((h, w), dtype=torch.float32)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    y_embed = y_embed / h
    x_embed = x_embed / w

    coords = torch.stack([x_embed, y_embed], dim=-1)  # coords are in [0, 1]^2 square and have (h, w, 2) shape
    coords = 2 * coords - 1
    coords = coords @ torch.randn((2, half_embed_dim))
    coords = 2 * np.pi * coords
    # outputs h x w x C shape
    pe = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    return pe.permute(2, 0, 1).unsqueeze(0)  # 1 x C x H x W

def get_edge_prediction(edge_prob: torch.Tensor, valid_lengths: Union[List[int], torch.Tensor]) -> Tuple[List, List]:
    if len(edge_prob.shape) == 3:
        # edge_prob is [l, l, 7]
        edge_prob = edge_prob.unsqueeze(0)
    assert len(edge_prob.shape) == 4 and edge_prob.shape[1] == edge_prob.shape[2] and edge_prob.shape[3] == 7,\
        f'edge_prob shape is {edge_prob.shape}, expected [b, l, l, 7] 4D tensor'
    b, l, _, d = edge_prob.shape
    if l == 0:
        return [], []
    for i in range(l):
        for j in range(i + 1, l):
            for k in range(5):
                edge_prob[:, i, j, k] = (edge_prob[:, i, j, k] + edge_prob[:, j, i, k]) / 2
                edge_prob[:, j, i, k] = edge_prob[:, i, j, k]
            edge_prob[:, i, j, 5] = (edge_prob[:, i, j, 5] + edge_prob[:, j, i, 6]) / 2
            edge_prob[:, i, j, 6] = (edge_prob[:, i, j, 6] + edge_prob[:, j, i, 5]) / 2
            edge_prob[:, j, i, 5] = edge_prob[:, i, j, 6]
            edge_prob[:, j, i, 6] = edge_prob[:, i, j, 5]
    score, prediction = torch.max(edge_prob, axis=-1)   # [b, l, l]
    return [prediction[i, :int(valid_lengths[i]), :int(valid_lengths[i])].tolist() for i in range(b)], \
           [score[i, :int(valid_lengths[i]), :int(valid_lengths[i])].tolist() for i in range(b)]

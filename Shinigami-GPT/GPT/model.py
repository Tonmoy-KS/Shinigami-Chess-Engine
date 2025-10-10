# gpt/model.py
import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig
from .kernels import HAS_TRITON, fused_swiglu_kernel

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RoPECache(nn.Module):
    def __init__(self, head_dim, max_seq_len, cfg: ModelConfig, device=None):
        super().__init__()
        base = 10000.0
        if cfg.rope_scaling:
            scaling_type = cfg.rope_scaling.type
            scaling_factor = cfg.rope_scaling.factor
            if scaling_type == 'linear':
                base /= scaling_factor
            elif scaling_type == 'ntk':
                base *= scaling_factor ** (head_dim / (head_dim - 2))
        
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    def forward(self, T, device):
        return self.cos_cached[:, :, :T, :].to(device), self.sin_cached[:, :, :T, :].to(device)

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_len, n_kv_heads, head_dim, sliding_window, device, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_kv_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, device=device, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, device=device, dtype=dtype))
        self.seq_len = 0
        self.sliding_window = sliding_window

    def update(self, k, v):
        B, _, T, _ = k.shape
        start_pos = self.seq_len % self.k_cache.shape[2]
        self.k_cache[:B, :, start_pos : start_pos + T] = k
        self.v_cache[:B, :, start_pos : start_pos + T] = v
        self.seq_len += T

        if self.seq_len > self.sliding_window:
            k_ret = self.k_cache[:, :, -self.sliding_window:]
            v_ret = self.v_cache[:, :, -self.sliding_window:]
        else:
            k_ret = self.k_cache[:, :, :self.seq_len]
            v_ret = self.v_cache[:, :, :self.seq_len]
        return k_ret, v_ret

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.n_embed // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.use_bias)
        self.k_proj = nn.Linear(cfg.n_embed, self.n_kv_heads * self.head_dim, bias=cfg.use_bias)
        self.v_proj = nn.Linear(cfg.n_embed, self.n_kv_heads * self.head_dim, bias=cfg.use_bias)
        self.o_proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.use_bias)

    def forward(self, x, rope: RoPECache, kv_cache: Optional[KVCache] = None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = rope(T, x.device)
        q = (q * cos) + (torch.cat((-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]), dim=-1) * sin)
        k = (k * cos) + (torch.cat((-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]), dim=-1) * sin)
        
        use_swa = kv_cache is not None and self.cfg.sliding_window_size > 0 and self.cfg.sliding_window_size < self.cfg.block_size
        if kv_cache is not None: k, v = kv_cache.update(k, v)

        if self.n_kv_heads != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
        
        is_causal = T > 1 and not use_swa
        
        if HAS_FLASH_ATTN:
            output = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=is_causal, 
                                     window_size=(-1, self.cfg.sliding_window_size) if use_swa else (-1, -1))
        else:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        
        y = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class FusedSwiGLU(nn.Module):
    def __init__(self, n_embed, hidden_dim, use_bias):
        super().__init__()
        self.w1 = nn.Linear(n_embed, hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(n_embed, hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(hidden_dim, n_embed, bias=use_bias)
    def forward(self, x):
        if HAS_TRITON and x.is_cuda and x.dim() == 3: # Triton kernel requires 3D tensor
            B, T, C = x.shape
            x_flat = x.view(-1, C)
            gate = self.w1(x_flat)
            content = self.w2(x_flat)
            fused_out = torch.empty_like(gate)
            grid = lambda META: (triton.cdiv(gate.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(gate.shape[1], META['BLOCK_SIZE_N']),)
            fused_swiglu_kernel[grid](x_flat, self.w1.weight.t(), fused_out, self.w2.weight.t(), 
                                      x_flat.shape[0], gate.shape[1], x_flat.shape[1],
                                      *x_flat.stride(), *self.w1.weight.t().stride(), *fused_out.stride(), *self.w2.weight.t().stride())
            return self.w3(fused_out).view(B, T, C)
        else:
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

class FFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden_dim = int(2/3 * 4 * cfg.n_embed)
        self.ffn_layer = FusedSwiGLU(cfg.n_embed, hidden_dim, cfg.use_bias)
    def forward(self, x): return self.ffn_layer(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.ffn = FFN(cfg)
        self.norm1 = RMSNorm(cfg.n_embed)
        self.norm2 = RMSNorm(cfg.n_embed)
    def forward(self, x, rope, kv_cache: Optional[KVCache] = None):
        x = x + self.attn(self.norm1(x), rope, kv_cache)
        x = x + self.ffn(self.norm2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embed)
        if cfg.untie_weights: self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
        else: self.lm_head = lambda x: F.linear(x, self.tok_embeddings.weight)
        self.rope_cache = RoPECache(cfg.n_embed // cfg.n_head, cfg.block_size, cfg)
        self.kv_caches: Optional[List[KVCache]] = None

    def setup_caches(self, max_batch_size, device, use_swa=False):
        max_seq_len = self.cfg.sliding_window_size if use_swa else self.cfg.block_size
        self.kv_caches = [
            KVCache(max_batch_size, max_seq_len, self.cfg.n_kv_heads, self.cfg.n_embed // self.cfg.n_head, 
                    self.cfg.sliding_window_size if use_swa else max_seq_len, device, self.tok_embeddings.weight.dtype)
            for _ in range(self.cfg.n_layer)
        ]

    def forward(self, idx, targets=None, loss_mask=None):
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            cache = self.kv_caches[i] if self.kv_caches is not None else None
            x = layer(x, self.rope_cache, cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if loss_mask is not None:
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                mask_flat = loss_mask.view(-1)
                loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat], ignore_index=-1)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
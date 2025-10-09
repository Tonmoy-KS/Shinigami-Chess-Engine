# model.py
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from config import ModelConfig

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class RoPECache(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, T):
        return self.cos_cached[:, :, :T, :], self.sin_cached[:, :, :T, :]

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_len, n_kv_heads, head_dim, device=None, dtype=None):
        super().__init__()
        self.cache_k = torch.zeros((max_batch_size, n_kv_heads, max_seq_len, head_dim), device=device, dtype=dtype)
        self.cache_v = torch.zeros((max_batch_size, n_kv_heads, max_seq_len, head_dim), device=device, dtype=dtype)
        self.seq_len = 0

    def update(self, k, v):
        B, n_kv_heads, T, head_dim = k.shape
        self.cache_k[:B, :, self.seq_len : self.seq_len + T, :] = k
        self.cache_v[:B, :, self.seq_len : self.seq_len + T, :] = v
        self.seq_len += T
        return self.cache_k[:B, :, :self.seq_len, :], self.cache_v[:B, :, :self.seq_len, :]

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.n_embed // cfg.n_head
        
        self.q_proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.k_proj = nn.Linear(cfg.n_embed, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embed, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)

    def forward(self, x, rope, kv_cache: Optional[KVCache] = None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = rope(T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        if self.n_kv_heads != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
        
        if self.cfg.use_flash_attn and flash_attn_func is not None and T > 1:
            output = flash_attn_func(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True)
        else:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        
        y = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class SwiGLU(nn.Module):
    def __init__(self, n_embed, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2/3 * 4 * n_embed)
        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embed, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MixtureOfExperts(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts = cfg.moe.num_experts
        self.num_experts_per_tok = cfg.moe.num_experts_per_tok
        self.gate = nn.Linear(cfg.n_embed, self.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg.n_embed) for _ in range(self.num_experts)])

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x = x.view(-1, C)
        router_logits = self.gate(x)
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(x.dtype)
        final_output = torch.zeros_like(x)
        
        flat_expert_indices = selected_experts.view(-1)
        flat_token_indices = torch.arange(x.size(0), device=x.device).repeat_interleave(self.num_experts_per_tok)
        
        for i in range(self.num_experts):
            expert_mask = (flat_expert_indices == i)
            if expert_mask.any():
                tok_indices = flat_token_indices[expert_mask]
                expert_input = x[tok_indices]
                expert_output = self.experts[i](expert_input)
                weights = routing_weights.view(-1)[expert_mask]
                final_output.index_add_(0, tok_indices, expert_output * weights.unsqueeze(1))

        return final_output.view(B, T, C)

class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.moe and cfg.moe.num_experts > 0:
            self.ffn_layer = MixtureOfExperts(cfg)
        else:
            self.ffn_layer = SwiGLU(cfg.n_embed)

    def forward(self, x):
        return self.ffn_layer(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
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
    def __init__(self, cfg_or_model_cfg):
        super().__init__()
        # Handle both ModelConfig and simplified draft config dicts
        if isinstance(cfg_or_model_cfg, ModelConfig):
            self.cfg = cfg_or_model_cfg
        else:
            @dataclass
            class SimpleConfig:
                __annotations__ = cfg_or_model_cfg
            self.cfg = SimpleConfig
            for k, v in cfg_or_model_cfg.items():
                setattr(self.cfg, k, v)

        self.tok_embeddings = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embed)
        self.layers = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(self.cfg.n_layer)])
        self.norm = RMSNorm(self.cfg.n_embed)
        
        if self.cfg.untie_weights:
            self.lm_head = nn.Linear(self.cfg.n_embed, self.cfg.vocab_size, bias=False)
        else:
            self.lm_head = self.tok_embeddings
        
        self.rope_cache = RoPECache(self.cfg.n_embed // self.cfg.n_head, self.cfg.block_size)
        self.kv_caches: List[KVCache] = []

    def setup_kv_cache(self, max_batch_size, max_seq_len, device, dtype):
        self.kv_caches = [
            KVCache(max_batch_size, max_seq_len, self.cfg.n_kv_heads, self.cfg.n_embed // self.cfg.n_head, device, dtype)
            for _ in range(self.cfg.n_layer)
        ]

    def clear_kv_cache(self):
        for cache in self.kv_caches:
            cache.seq_len = 0

    def forward(self, idx, targets=None, use_kv_cache=False):
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            kv_cache = self.kv_caches[i] if use_kv_cache else None
            x = layer(x, self.rope_cache, kv_cache)
        x = self.norm(x)
        
        if self.cfg.untie_weights:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.tok_embeddings.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        self.setup_kv_cache(max_batch_size=idx.size(0), max_seq_len=self.cfg.block_size, device=idx.device, dtype=self.tok_embeddings.weight.dtype)

        # Process the prompt
        _, _ = self(idx, use_kv_cache=True)

        # Generate new tokens
        for _ in range(max_new_tokens):
            next_token_idx = idx[:, -1:]
            logits, _ = self(next_token_idx, use_kv_cache=True)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.clear_kv_cache()
        self.train()
        return idx

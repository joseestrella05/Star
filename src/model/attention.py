"""Token/positional embeddings and multi-head causal self-attention."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StarConfig


class Embeddings(nn.Module):
    """Learned token + absolute positional embeddings: h_0 = E[x] + P[0..T-1]."""

    def __init__(self, config: StarConfig):
        super().__init__()
        self.block_size = config.block_size
        self.tok = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        T = idx.size(1)
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        return self.drop(self.tok(idx) + self.pos(pos))


class CausalSelfAttention(nn.Module):
    """softmax(Q K^T / sqrt(d_k) + M) V per head, with M masking future positions."""

    def __init__(self, config: StarConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = config.flash
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)  # fused Q, K, V
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            y = self.attn_dropout(F.softmax(att, dim=-1)) @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads
        return self.resid_dropout(self.c_proj(y))

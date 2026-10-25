"""StarGPT: a decoder-only Transformer language model."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention, Embeddings
from .config import StarConfig


class FeedForward(nn.Module):
    """Position-wise MLP: W2 GELU(W1 x), expanding to 4 * n_embd."""

    def __init__(self, config: StarConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class TransformerBlock(nn.Module):
    """Pre-LN block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, config: StarConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.ffn(self.ln_2(x))


def filter_logits(logits, top_k=None, top_p=None):
    """Mask logits outside the top-k set and/or the smallest nucleus with mass >= top_p."""
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        # Drop a token once the mass *before* it already reaches top_p (always keeps the best one).
        remove = probs.cumsum(dim=-1) - probs >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


class StarGPT(nn.Module):
    def __init__(self, config: StarConfig):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.tok.weight  # weight tying

        self.apply(self._init_weights)
        # GPT-2: scale residual output projections so activations don't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self, non_embedding=True):
        """Parameter count; tied lm_head is counted once. Positional table excluded by default."""
        n = sum(p.numel() for p in self.parameters())
        return n - self.embeddings.pos.weight.numel() if non_embedding else n

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None  # inference only needs the last position
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_id=None):
        """Autoregressive sampling. temperature=0 means greedy decoding."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size:])
            logits = logits[:, -1, :]
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = filter_logits(logits / temperature, top_k, top_p)
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx

"""Token-stream dataset producing (X, Y) windows shifted by one position."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_tokens(corpus_path, tokenizer, cache_path=None):
    """Encode the corpus once and cache the ids as a uint16 .bin file."""
    assert tokenizer.vocab_size <= 2**16, "uint16 cache requires vocab_size <= 65536"
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists() and cache.stat().st_mtime > Path(corpus_path).stat().st_mtime:
        return np.fromfile(cache, dtype=np.uint16)
    ids = np.array(tokenizer.encode(Path(corpus_path).read_text(encoding="utf-8")), dtype=np.uint16)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        ids.tofile(cache)
    return ids


class TokenDataset(Dataset):
    """Window i: X = t[i : i+T], Y = t[i+1 : i+T+1] (next-token targets)."""

    def __init__(self, tokens, block_size):
        assert len(tokens) > block_size, "not enough tokens for one block"
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, i):
        chunk = self.tokens[i: i + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def make_loaders(tokens, block_size, batch_size, val_frac=0.1):
    """Contiguous split (val = tail) so validation text never overlaps training windows."""
    n = int(len(tokens) * (1 - val_frac))
    train, val = TokenDataset(tokens[:n], block_size), TokenDataset(tokens[n:], block_size)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=True))

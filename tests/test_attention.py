import torch

from src.model.attention import CausalSelfAttention, Embeddings
from src.model.config import StarConfig

CFG = dict(vocab_size=64, block_size=16, n_embd=32, n_head=4, dropout=0.0)


def test_embeddings_shape():
    emb = Embeddings(StarConfig(**CFG))
    assert emb(torch.randint(0, 64, (2, 10))).shape == (2, 10, 32)


def test_flash_matches_masked_path():
    torch.manual_seed(0)
    flash = CausalSelfAttention(StarConfig(**CFG, flash=True)).eval()
    manual = CausalSelfAttention(StarConfig(**CFG, flash=False)).eval()
    manual.load_state_dict(flash.state_dict())
    x = torch.randn(2, 16, 32)
    assert torch.allclose(flash(x), manual(x), atol=1e-5)


def test_causality():
    torch.manual_seed(0)
    for flash in (True, False):
        attn = CausalSelfAttention(StarConfig(**CFG, flash=flash)).eval()
        x = torch.randn(1, 12, 32)
        y = x.clone()
        y[:, 8:] = torch.randn(1, 4, 32)  # perturb only the future
        assert torch.allclose(attn(x)[:, :8], attn(y)[:, :8], atol=1e-6)

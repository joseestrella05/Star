import math

import torch

from src.model.config import StarConfig
from src.model.star_gpt import StarGPT, filter_logits

TINY = StarConfig(vocab_size=64, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)


def test_forward_loss_and_tying():
    torch.manual_seed(0)
    model = StarGPT(TINY)
    x, y = torch.randint(0, 64, (2, 2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 64)
    assert abs(loss.item() - math.log(64)) < 0.3  # near-uniform predictions at init
    assert model.lm_head.weight is model.embeddings.tok.weight


def test_base_config_param_count():
    n = StarGPT(StarConfig()).num_params(non_embedding=False)
    assert 5e6 < n < 7e6


def test_learns_tiny_sequence():
    torch.manual_seed(0)
    model = StarGPT(TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.arange(16).unsqueeze(0)
    for _ in range(60):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
    out = model.eval().generate(x[:, :4], 6, temperature=0)
    assert out.tolist() == [list(range(10))]


def test_filter_logits():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    assert torch.isinf(filter_logits(logits, top_k=2)[0, 2:]).all()
    nucleus = filter_logits(logits, top_p=0.75)  # 0.5 + 0.3 crosses 0.75
    assert torch.isfinite(nucleus[0, :2]).all() and torch.isinf(nucleus[0, 2:]).all()
    assert torch.isfinite(filter_logits(logits, top_p=0.01)[0, 0])  # best token always kept


def test_generate_sampling_and_eos():
    torch.manual_seed(0)
    model = StarGPT(TINY).eval()
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), 40, temperature=0.8, top_k=10, top_p=0.9)
    assert out.shape == (1, 41)  # longer than block_size: context is cropped
    greedy_next = model.generate(torch.zeros(1, 1, dtype=torch.long), 1, temperature=0)[0, 1].item()
    stopped = model.generate(torch.zeros(1, 1, dtype=torch.long), 10, temperature=0, eos_id=greedy_next)
    assert stopped.shape == (1, 2)

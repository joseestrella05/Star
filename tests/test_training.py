import numpy as np
import torch

from src.model.config import StarConfig
from src.model.star_gpt import StarGPT
from src.training.dataset import TokenDataset, make_loaders
from src.training.trainer import TrainConfig, Trainer, get_lr


def test_shifted_windows():
    ds = TokenDataset(np.arange(10, dtype=np.uint16), block_size=4)
    assert len(ds) == 6
    x, y = ds[2]
    assert x.tolist() == [2, 3, 4, 5] and y.tolist() == [3, 4, 5, 6]


def test_lr_schedule():
    cfg = TrainConfig(max_iters=100, lr=1.0, min_lr=0.1, warmup_iters=10)
    assert get_lr(0, cfg) == 0.1 and get_lr(9, cfg) == 1.0
    assert abs(get_lr(55, cfg) - 0.55) < 1e-9  # cosine midpoint
    assert abs(get_lr(100, cfg) - 0.1) < 1e-9


def test_trainer_reduces_loss_and_checkpoints(tmp_path):
    torch.manual_seed(0)
    tokens = np.tile(np.arange(32, dtype=np.uint16), 200)  # perfectly predictable stream
    train, val = make_loaders(tokens, block_size=16, batch_size=8)
    model = StarGPT(StarConfig(vocab_size=32, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0))
    cfg = TrainConfig(max_iters=60, lr=3e-3, min_lr=3e-4, warmup_iters=5, eval_interval=30, eval_iters=2,
                      log_interval=1000, out_dir=str(tmp_path))
    best = Trainer(model, train, val, cfg, device="cpu").train()
    assert best < 2.0  # init loss is ln(32) ~ 3.47
    ckpt = torch.load(tmp_path / "best.pt")
    assert ckpt["model_config"]["n_embd"] == 32 and ckpt["val_loss"] == best

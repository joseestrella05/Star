"""Training loop: AdamW, warmup + cosine LR, gradient clipping, checkpointing."""
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class TrainConfig:
    max_iters: int = 5000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 50
    out_dir: str = "checkpoints"


def get_lr(it, cfg: TrainConfig):
    """Linear warmup to cfg.lr, then cosine decay to cfg.min_lr at cfg.max_iters."""
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    progress = min(1.0, (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters))
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def make_optimizer(model, cfg: TrainConfig):
    # Decay matrices only; biases and LayerNorm gains stay unregularized.
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def infinite(loader):
    while True:
        yield from loader


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg: TrainConfig, device=None, extra_state=None):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.cfg = cfg
        self.optimizer = make_optimizer(model, cfg)
        self.extra_state = extra_state or {}  # e.g. tokenizer path, stored in checkpoints
        self.best_val = float("inf")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        out = {}
        for split, loader in (("train", self.train_loader), ("val", self.val_loader)):
            batches, losses = infinite(loader), []
            for _ in range(self.cfg.eval_iters):
                x, y = (t.to(self.device) for t in next(batches))
                losses.append(self.model(x, y)[1].item())
            out[split] = sum(losses) / len(losses)
        self.model.train()
        return out

    def save(self, it, val_loss, name="best.pt"):
        path = Path(self.cfg.out_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "model_config": asdict(self.model.config),
            "train_config": asdict(self.cfg),
            "iter": it,
            "val_loss": val_loss,
            **self.extra_state,
        }, path)
        return path

    def train(self):
        cfg, batches = self.cfg, infinite(self.train_loader)
        self.model.train()
        t0 = time.time()
        for it in range(cfg.max_iters + 1):
            if it % cfg.eval_interval == 0 or it == cfg.max_iters:
                losses = self.evaluate()
                ppl = math.exp(min(losses["val"], 20))
                print(f"[eval] iter {it}: train {losses['train']:.4f} | val {losses['val']:.4f} | val ppl {ppl:.2f}")
                if losses["val"] < self.best_val:
                    self.best_val = losses["val"]
                    print(f"[ckpt] saved {self.save(it, self.best_val)}")
            if it == cfg.max_iters:
                break

            lr = get_lr(it, cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            x, y = (t.to(self.device) for t in next(batches))
            _, loss = self.model(x, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            if it % cfg.log_interval == 0:
                dt, t0 = time.time() - t0, time.time()
                print(f"iter {it}: loss {loss.item():.4f} | lr {lr:.2e} | grad norm {norm:.2f} | {dt:.1f}s")
        return self.best_val

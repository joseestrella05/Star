#!/usr/bin/env python3
"""Pre-train StarGPT on the tokenized business corpus."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.config import StarConfig  # noqa: E402
from src.model.star_gpt import StarGPT  # noqa: E402
from src.tokenizer.bpe import BPETokenizer  # noqa: E402
from src.training.dataset import load_tokens, make_loaders  # noqa: E402
from src.training.trainer import TrainConfig, Trainer, get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/business_corpus.txt"))
    ap.add_argument("--tokenizer", type=Path, default=Path("checkpoints/tokenizer.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/tokens.bin"))
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-iters", type=int, default=200)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (auto-detected by default)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = BPETokenizer.load(args.tokenizer)
    tokens = load_tokens(args.corpus, tok, args.cache)
    train_loader, val_loader = make_loaders(tokens, args.block_size, args.batch_size)

    model = StarGPT(StarConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_embd=args.n_embd,
                               n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout))
    device = args.device or get_device()
    print(f"device={device} | tokens={len(tokens):,} | params={model.num_params(non_embedding=False) / 1e6:.2f}M")

    cfg = TrainConfig(max_iters=args.max_iters, lr=args.lr, min_lr=args.lr / 10, warmup_iters=args.warmup_iters,
                      eval_interval=args.eval_interval, out_dir=args.out_dir)
    trainer = Trainer(model, train_loader, val_loader, cfg, device=device,
                      extra_state={"tokenizer": str(args.tokenizer)})
    best = trainer.train()
    print(f"done. best val loss {best:.4f}")


if __name__ == "__main__":
    main()

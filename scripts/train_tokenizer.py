#!/usr/bin/env python3
"""Train the Star byte-level BPE tokenizer on the business corpus."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tokenizer.bpe import BPETokenizer  # noqa: E402

TERMS = ["EBITDA", "ROI", "YoY", "CAPEX", "OPEX", "Q1", "Q2", "Q3", "Q4"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/business_corpus.txt"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/tokenizer.json"))
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--max-chars", type=int, default=None, help="train on the first N chars only")
    args = ap.parse_args()

    text = args.input.read_text(encoding="utf-8")
    sample = text[: args.max_chars]
    tok = BPETokenizer()
    t0 = time.time()
    tok.train(sample, args.vocab_size, verbose=True)
    print(f"trained vocab_size={tok.vocab_size} on {len(sample):,} chars in {time.time() - t0:.1f}s")

    probe = text[-200_000:]
    ids = tok.encode(probe)
    assert tok.decode(ids) == probe, "round-trip failed"
    print(f"compression on held-out text: {len(probe.encode('utf-8')) / len(ids):.2f} bytes/token")
    for t in TERMS:
        print(f"  {' ' + t!r:>10} -> {tok.encode(' ' + t)}")

    tok.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

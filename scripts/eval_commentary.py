#!/usr/bin/env python3
"""Evaluate report commentary: does the model's closing match the metrics, and does it pass verification?

Draws random metric sets balanced across the three closing cases, runs the real report engine and compares
the model's commentary with the rule-based closing for those figures.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.api.server import create_app, load_model  # noqa: E402
from src.reports.engine import generate_report  # noqa: E402
from src.reports.schemas import ModelInfo, ReportRequest  # noqa: E402
from src.reports.templates import CASH_FLOW, closing_index  # noqa: E402
from src.tokenizer.bpe import BPETokenizer  # noqa: E402


def random_metrics(rng, target):
    """Metric set (thousands) whose rule-based closing is `target`."""
    while True:
        rev = rng.uniform(50_000, 5_000_000)
        ebitda = rev * rng.uniform(0.1, 0.45)
        capex = rev * rng.uniform(0.03, 0.15)
        ocf = ebitda * rng.uniform(0.4, 0.95)
        net_debt = ebitda * rng.uniform(-0.3, 5.0)
        if closing_index(net_debt / ebitda, ocf - capex) == target:
            r = lambda x: round(x, 2)  # noqa: E731
            return dict(revenue=r(rev), ebitda=r(ebitda), operating_cash_flow=r(ocf), capex=r(capex),
                        working_capital_change=r(rev * rng.uniform(-0.03, 0.05)),
                        cash=r(rev * rng.uniform(0.05, 0.4)), net_debt=r(net_debt))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--samples", type=int, default=90)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, _ = load_model(args.checkpoint, args.device)
    tok = BPETokenizer.load(args.tokenizer)
    complete = create_app(model, tok, device=args.device).state.star["complete"]
    info = ModelInfo(checkpoint_sha256="eval", params=model.num_params(non_embedding=False))
    rng = random.Random(args.seed)

    stats = Counter()
    for i in range(args.samples):
        target, lang = i % 3, rng.choice(("en", "es"))
        req = ReportRequest(company_name="Eval Co", period="Q2 2026", language=lang,
                            metrics=random_metrics(rng, target), generation={"seed": i, "max_attempts": 1})
        report = generate_report(req, complete, info)
        commentary = report.sections[-1]
        expected = CASH_FLOW[lang]["closings"][target]
        stats["verified"] += report.verification.passed
        stats["published_correct"] += commentary.markdown == expected  # model or fallback: what the client reads
        stats[f"correct_{target}"] += commentary.source == "model" and commentary.markdown == expected
        stats[f"total_{target}"] += 1

    n = args.samples
    correct = sum(stats[f"correct_{t}"] for t in range(3))
    print(f"samples: {n}")
    print(f"model closing matches the metrics:     {correct}/{n} ({correct / n:.0%}; chance = 33%)")
    print(f"model closing accepted (figures+rule): {stats['verified']}/{n} ({stats['verified'] / n:.0%})")
    print(f"published closing correct (w/ guard):  {stats['published_correct']}/{n} "
          f"({stats['published_correct'] / n:.0%})")
    names = ["adequate liquidity", "deleveraging", "working capital"]
    for t in range(3):
        print(f"  {names[t]:<20} {stats[f'correct_{t}']}/{stats[f'total_{t}']}")


if __name__ == "__main__":
    main()

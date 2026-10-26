"""Report engine: canonical prompt -> model commentary -> figure verification -> Markdown."""
from datetime import datetime, timezone
from uuid import uuid4

from .schemas import ReportResponse, ReportSection, Verification
from .templates import build_cash_flow
from .verify import filter_unverified

COMMENTARY_TOKENS = 60  # the corpus commentary is one sentence (~20 tokens)


def generate_report(req, complete, model_info, currency_label=""):
    """`complete(prompt, max_new_tokens, temperature, seed) -> str` returns the raw continuation."""
    canonical = build_cash_flow(req, currency_label)
    opts = req.generation
    commentary, rejected, attempts = "", [], 0
    for attempt in range(opts.max_attempts):
        attempts += 1
        seed = None if opts.seed is None else opts.seed + attempt
        text = complete(canonical.prompt, COMMENTARY_TOKENS, opts.temperature, seed)
        first_line = text.strip().split("\n", 1)[0]  # commentary is a single line in the corpus layout
        kept, bad = filter_unverified(first_line, canonical.prompt)
        rejected.extend(bad)
        if kept and not bad:
            commentary = kept
            break
    source = "model" if commentary else "fallback"
    commentary = commentary or canonical.fallback

    sections = [
        ReportSection(key="header", markdown=f"# {canonical.title}", source="computed"),
        ReportSection(key="table", markdown=canonical.table, source="computed"),
        ReportSection(key="analysis", markdown=canonical.body.replace("\n", " "), source="computed"),
        ReportSection(key="commentary", markdown=commentary, source=source),
    ]
    return ReportResponse(
        report_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        request=req,
        markdown="\n\n".join(s.markdown for s in sections) + "\n",
        sections=sections,
        verification=Verification(passed=source == "model", attempts=attempts, rejected_figures=rejected),
        model=model_info,
    )

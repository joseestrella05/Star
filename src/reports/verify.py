"""Figure verification: every number in generated text must appear in the computed reference."""
import regex as re

# Same figure shape the tokenizer keeps as one chunk: "$1,234.5M", "-$3.2", "12,3%", "1.5x", "2026".
FIGURE = re.compile(r"-?\$?\p{N}+(?:[.,]\p{N}+)*(?:%|[MBKx](?!\p{L}))?")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def figures(text):
    return FIGURE.findall(text)


def filter_unverified(text, reference):
    """Drop sentences containing any figure absent from `reference`.

    Returns (kept_text, rejected_figures). Matching is on the exact formatted string, so a
    rounded, re-scaled or reformatted number counts as unverified.
    """
    allowed = set(figures(reference))
    kept, rejected = [], []
    for sentence in SENTENCE_END.split(text.strip()):
        bad = [f for f in figures(sentence) if f not in allowed]
        if bad:
            rejected.extend(bad)
        elif sentence:
            kept.append(sentence)
    return " ".join(kept), rejected

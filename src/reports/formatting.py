"""Financial number formatting shared by the training corpus and report prompts.

Prompts must match the corpus format exactly, so both import from here.
"""


def num(x, lang, d=1):
    s = f"{x:,.{d}f}"
    return s.translate(str.maketrans(",.", ".,")) if lang == "es" else s


def money(x, lang):
    """Amounts are in USD millions."""
    sign = "-" if x < 0 else ""
    return f"{sign}${num(abs(x), lang)}M" if lang == "en" else f"{sign}${num(abs(x), lang)} M"


def pct(x, lang):
    return f"{num(x, lang)}%"

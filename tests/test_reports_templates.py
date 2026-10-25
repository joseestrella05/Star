from scripts.generate_business_corpus import generate_corpus
from src.reports.schemas import ReportRequest
from src.reports.templates import build_cash_flow
from src.reports.verify import figures, filter_unverified

METRICS = dict(revenue=1200, ebitda=300, operating_cash_flow=250, capex=100, working_capital_change=-20,
               cash=80, net_debt=450)


def req(lang="en", **metrics):
    return ReportRequest(company_name="Atlas Logistics", period="Q2 2026", language=lang,
                         metrics={**METRICS, **metrics})


def test_prompt_matches_corpus_layout():
    r = build_cash_flow(req())
    assert r.prompt.startswith("Atlas Logistics - Cash Flow Analysis, Q2 2026\n\n")
    assert "free cash flow (FCF) of $150.0M" in r.body
    assert "a leverage ratio of 1.5x net debt/EBITDA." in r.body
    assert "working capital change of -$20.0M" in r.body
    assert r.prompt.endswith(".\n")
    # The same layout (title, blank line, sentences) exists verbatim in the training corpus.
    corpus = "\n".join(generate_corpus(300, seed=1))
    assert "Cash Flow Analysis, Q" in corpus and "FCF conversion, measured as FCF over EBITDA, was" in corpus


def test_spanish_formatting():
    r = build_cash_flow(req("es", revenue=1234.5, capex=123.45))
    assert "Análisis de Flujo de Caja" in r.title
    assert "El CAPEX sumó $123,5 M (10,0% de los ingresos)" in r.body
    assert "| Ingresos | $1.234,5 M |" in r.table


def test_negative_ebitda_drops_ratio_sentences():
    r = build_cash_flow(req(ebitda=-10))
    assert "None" not in r.body and "FCF conversion" not in r.body and "leverage" not in r.body
    assert "| FCF conversion | n/a |" in r.table


def test_fallback_follows_metrics():
    assert "deleveraging" in build_cash_flow(req(net_debt=1500)).fallback
    assert "Liquidity remains adequate" in build_cash_flow(req()).fallback
    assert "working capital" in build_cash_flow(req(capex=400)).fallback  # negative FCF


def test_verification_keeps_only_grounded_sentences():
    ref = build_cash_flow(req()).prompt
    text = ("Liquidity remains adequate. Free cash flow reached $150.0M. "
            "Revenue grew 12.5% YoY. Net debt was $450M.")
    kept, rejected = filter_unverified(text, ref)
    assert kept == "Liquidity remains adequate. Free cash flow reached $150.0M."
    assert rejected == ["12.5%", "$450M"]  # invented figure, and a correct value in the wrong format


def test_figure_extraction():
    assert figures("EBITDA of $1,234.5M, -$3.2 M, 12,3% and 1.5x in Q2 2026") == \
        ["$1,234.5M", "-$3.2", "12,3%", "1.5x", "2", "2026"]

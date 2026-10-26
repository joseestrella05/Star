from scripts.generate_business_corpus import generate_corpus
from src.reports.schemas import ReportRequest
from src.reports.templates import build_cash_flow
from src.reports.verify import figures, filter_unverified

# Thousands of the institution's currency.
METRICS = dict(revenue=1_200_000, ebitda=300_000, operating_cash_flow=250_000, capex=100_000,
               working_capital_change=-20_000, cash=80_000, net_debt=450_000)


def req(lang="en", **metrics):
    return ReportRequest(company_name="Atlas Logistics", period="Q2 2026", language=lang,
                         metrics={**METRICS, **metrics})


def test_prompt_matches_corpus_layout():
    r = build_cash_flow(req())
    assert r.prompt.startswith("Atlas Logistics - Cash Flow Analysis, Q2 2026\n\n")
    assert "free cash flow (FCF) of 150,000" in r.body
    assert "a leverage ratio of 1.5x net debt/EBITDA." in r.body
    assert "working capital change of -20,000" in r.body
    assert "$" not in r.prompt
    assert r.prompt.endswith(".\n")
    # The same layout (title, blank line, sentences) exists verbatim in the training corpus.
    corpus = "\n".join(generate_corpus(300, seed=1))
    assert "Cash Flow Analysis, Q" in corpus and "FCF conversion, measured as FCF over EBITDA, was" in corpus


def test_spanish_formatting():
    r = build_cash_flow(req("es", revenue=1_234_500.4, capex=123_450.5), currency_label="DOP")
    assert "Análisis de Flujo de Caja" in r.title
    assert "El CAPEX sumó 123.451 (10,0% de los ingresos)" in r.body
    assert "| Métrica | Valor (miles de DOP) |" in r.table and "| Ingresos | 1.234.500 |" in r.table
    assert "DOP" not in r.prompt  # the label is display-only; the model never sees a currency


def test_table_header_without_label():
    assert "| Metric | Value (thousands) |" in build_cash_flow(req()).table


def test_negative_ebitda_drops_ratio_sentences():
    r = build_cash_flow(req(ebitda=-10_000))
    assert "None" not in r.body and "FCF conversion" not in r.body and "leverage" not in r.body
    assert "| FCF conversion | n/a |" in r.table


def test_fallback_follows_metrics():
    assert "deleveraging" in build_cash_flow(req(net_debt=1_500_000)).fallback
    assert "Liquidity remains adequate" in build_cash_flow(req()).fallback
    assert "working capital" in build_cash_flow(req(capex=400_000)).fallback  # negative FCF


def test_corpus_closings_follow_the_same_rule():
    # Every cash-flow doc in the corpus ends with the closing the rule picks for its own figures.
    from scripts.generate_business_corpus import CASH_FLOW
    closings = {c: i for t in CASH_FLOW.values() for i, c in enumerate(t["closings"])}
    docs = [d for d in generate_corpus(400, seed=3) if "Cash Flow Analysis" in d or "Flujo de Caja" in d]
    seen = {closings[d.rsplit("\n", 1)[1]] for d in docs}
    assert seen == {0, 1, 2}  # all three outcomes occur, so the model sees each case


def test_verification_keeps_only_grounded_sentences():
    ref = build_cash_flow(req()).prompt
    text = ("Liquidity remains adequate. Free cash flow reached 150,000. "
            "Revenue grew 12.5% YoY. Net debt was 450.0.")
    kept, rejected = filter_unverified(text, ref)
    assert kept == "Liquidity remains adequate. Free cash flow reached 150,000."
    assert rejected == ["12.5%", "450.0"]  # invented figure, and a value in the wrong scale/format


def test_figure_extraction():
    assert figures("EBITDA of $1,234.5M, -$3.2 M, 12,3% and 1.5x in Q2 2026") == \
        ["$1,234.5M", "-$3.2", "12,3%", "1.5x", "2", "2026"]

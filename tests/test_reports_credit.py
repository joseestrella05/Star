from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from src.api.server import create_app
from src.reports.credit import DEFAULT_POLICY, annual_debt_service, assess, max_principal
from src.reports.schemas import AnyReportRequest, CreditAnalysisRequest, ReportRequest
from src.reports.templates import build_credit
from src.reports.verify import filter_unverified
from tests.test_api import MODEL, TOK

# Thousands. Base case is favorable: DSCR ~2.11x, leverage ~2.0x, coverage 1.12x.
FINANCIALS = dict(revenue=1_200_000, ebitda=240_000, operating_cash_flow=190_000, maintenance_capex=40_000,
                  total_debt=350_000, cash=60_000, existing_debt_service=20_000)
FACILITY = dict(amount=200_000, annual_rate_pct=10, term_months=60)
MORTGAGE = dict(type="mortgage", description="Nave industrial", appraised_value=320_000)


def credit(lang="es", financials=None, facility=None, collateral=None):
    return CreditAnalysisRequest(
        report_type="credit_analysis", company_name="Constructora Caribe", period="Q2 2026", language=lang,
        financials={**FINANCIALS, **(financials or {})}, facility={**FACILITY, **(facility or {})},
        collateral=[MORTGAGE] if collateral is None else collateral)


# --- arithmetic ---------------------------------------------------------------------------------------------

def test_french_annuity_matches_reference_value():
    # 1,000,000 at 12% for 12 months: textbook monthly payment 88,848.79.
    annual = annual_debt_service(Decimal(1_000_000), Decimal(12), 12)
    assert abs(annual / 12 - Decimal("88848.79")) < Decimal("0.01")


def test_zero_rate_is_straight_line():
    assert annual_debt_service(Decimal(120_000), Decimal(0), 24) == Decimal(60_000)
    assert max_principal(Decimal(60_000), Decimal(0), 24) == Decimal(120_000)


@pytest.mark.parametrize("rate,term", [(Decimal("12.5"), 60), (Decimal("7.25"), 240), (Decimal(0), 36)])
def test_max_principal_inverts_debt_service(rate, term):
    service = annual_debt_service(Decimal(345_678), rate, term)
    assert abs(max_principal(service, rate, term) - 345_678) < Decimal("0.000001")


def test_haircuts_by_collateral_type():
    collateral = [dict(type="cash_deposit", description="Certificado", appraised_value=100_000),
                  dict(type="mortgage", description="Local", appraised_value=100_000),
                  dict(type="pledge", description="Maquinaria", appraised_value=100_000),
                  dict(type="personal_guarantee", description="Fianza socios", appraised_value=900_000)]
    a = assess(credit(collateral=collateral))
    assert a.adjusted_collateral == Decimal(100_000 + 70_000 + 50_000 + 0)


# --- decisions ----------------------------------------------------------------------------------------------

def test_favorable():
    a = assess(credit())
    assert a.cfads == Decimal(150_000)
    assert a.leverage_after == Decimal(490_000) / Decimal(240_000)
    assert a.coverage == Decimal("1.12") and a.collateral_gap == 0
    assert a.dscr > DEFAULT_POLICY.min_dscr and a.decision == "favorable" and a.conditions == []


def test_conditional_reduce_amount_and_collateral():
    a = assess(credit(facility={"amount": 420_000, "annual_rate_pct": "12.5"}))
    assert DEFAULT_POLICY.hard_min_dscr <= a.dscr < DEFAULT_POLICY.min_dscr
    assert a.decision == "conditional" and a.conditions == ["reduce_amount", "add_collateral"]
    assert a.collateral_gap == Decimal(420_000 - 224_000)
    # Borrowing exactly the suggested maximum lands on the policy DSCR.
    at_max = assess(credit(facility={"amount": a.max_loan.quantize(Decimal("0.01")), "annual_rate_pct": "12.5"}))
    assert abs(at_max.dscr - DEFAULT_POLICY.min_dscr) < Decimal("0.0001")


def test_conditional_restructure_when_no_capacity_left():
    a = assess(credit(financials={"existing_debt_service": 125_000}, facility={"amount": 1_000}))
    assert a.max_loan == 0 and a.decision == "conditional" and a.conditions[0] == "restructure_debt"


def test_conditional_leverage_covenant():
    a = assess(credit(financials={"total_debt": 700_000}))  # (700k + 200k - 60k) / 240k = 3.5x
    assert a.decision == "favorable"
    a = assess(credit(financials={"total_debt": 800_000}))  # 3.9x: above 3.5x, below the 4.5x hard limit
    assert a.decision == "conditional" and a.conditions == ["leverage_covenant"]


@pytest.mark.parametrize("financials,failing", [
    ({"existing_debt_service": 110_000}, "dscr"),     # DSCR < 1.00x
    ({"total_debt": 1_000_000}, "leverage"),          # 4.8x > 4.5x
    ({"ebitda": -5_000}, "leverage"),                 # leverage not computable
])
def test_unfavorable(financials, failing):
    a = assess(credit(financials=financials))
    assert a.decision == "unfavorable" and a.conditions == []
    assert [c.key for c in a.criteria if c.hard_fail] == [failing]


# --- templates ----------------------------------------------------------------------------------------------

CASES = {
    "favorable": {},
    "conditional": {"facility": {"amount": 420_000, "annual_rate_pct": "12.5"}},
    "restructure": {"financials": {"existing_debt_service": 125_000}, "facility": {"amount": 1_000}},
    "unfavorable_dscr": {"financials": {"existing_debt_service": 110_000}},
    "unfavorable_ebitda": {"financials": {"ebitda": -5_000}},
}


@pytest.mark.parametrize("lang", ["es", "en"])
@pytest.mark.parametrize("case", CASES)
def test_every_figure_in_the_opinion_is_traceable(case, lang):
    """Auditability: each number the committee reads in the opinion appears in the computed sections."""
    r = build_credit(credit(lang, **CASES[case]), currency_label="DOP")
    _, untraceable = filter_unverified(r.opinion, "\n".join([r.title, r.table, r.analysis, r.criteria]))
    assert untraceable == []
    assert "$" not in r.opinion + r.analysis + r.table


def test_spanish_conditional_report():
    r = build_credit(credit("es", **CASES["conditional"]), currency_label="DOP")
    assert r.title == "Constructora Caribe - Análisis de Crédito, Q2 2026"
    assert "| Métrica | Valor (miles de DOP) |" in r.table and "| Tasa anual | 12,50% |" in r.table
    assert "| DSCR | 1,12x | ≥ 1,25x (mín. 1,00x) | No |" in r.criteria
    assert r.opinion.startswith("**Dictamen: FAVORABLE CON CONDICIONES.**")
    assert "- Reducir el monto a un máximo de" in r.opinion
    assert "- Constituir garantías adicionales con un valor ajustado de al menos 196.000." in r.opinion
    assert "requiere revisión y firma del comité de crédito" in r.opinion


def test_english_unfavorable_has_reason_and_counteroffer():
    r = build_credit(credit("en", **CASES["unfavorable_dscr"]))
    assert r.opinion.startswith("**Opinion: UNFAVORABLE.**")
    assert "cash flow does not cover debt service" in r.opinion
    assert "Alternative: a facility of up to" in r.opinion
    assert "| Metric | Value (thousands) |" in r.table


def test_no_counteroffer_without_capacity():
    r = build_credit(credit("en", financials={"ebitda": -5_000, "operating_cash_flow": 10_000}))
    assert "EBITDA is not positive." in r.opinion and "Alternative" not in r.opinion


# --- schema and endpoint ------------------------------------------------------------------------------------

ADAPTER = TypeAdapter(AnyReportRequest)


def test_discriminator_keeps_cash_flow_requests_working():
    legacy = dict(company_name="Atlas", period="Q2 2026",
                  metrics=dict(revenue=10, ebitda=2, operating_cash_flow=2, capex=1, cash=1, net_debt=1))
    assert isinstance(ADAPTER.validate_python(legacy), ReportRequest)  # no report_type -> cash flow
    req = ADAPTER.validate_python(credit().model_dump())
    assert isinstance(req, CreditAnalysisRequest)


@pytest.mark.parametrize("patch", [
    {"report_type": "mortgage_appraisal"},
    {"facility": {**FACILITY, "term_months": 0}},
    {"facility": {**FACILITY, "annual_rate_pct": 101}},
    {"facility": {**FACILITY, "amount": 0}},
    {"collateral": [{**MORTGAGE, "type": "crypto"}]},
    {"collateral": [{**MORTGAGE, "description": "<script>"}]},
    {"collateral": [MORTGAGE] * 21},
    {"financials": {**FINANCIALS, "ebitda": 2_000_000}},
    {"financials": {**FINANCIALS, "debt_service": 1}},
])
def test_rejects_invalid_credit_requests(patch):
    with pytest.raises(ValidationError):
        ADAPTER.validate_python({**credit().model_dump(mode="json"), **patch})


def test_endpoint_never_calls_the_model():
    app = create_app(MODEL, TOK, checkpoint_sha256="abc")

    def model_must_not_run(*args, **kwargs):
        raise AssertionError("credit analysis must not call the model")
    app.state.star["complete"] = model_must_not_run
    with TestClient(app) as c:
        body = c.post("/api/v1/reports/generate", json=credit().model_dump(mode="json")).json()
    assert [s["key"] for s in body["sections"]] == ["header", "table", "analysis", "criteria", "opinion"]
    assert {s["source"] for s in body["sections"]} == {"computed"}
    assert body["verification"] == {"passed": True, "attempts": 0, "rejected_figures": []}
    assert body["policy"]["min_dscr"] == "1.25" and body["policy"]["haircuts"]["mortgage"] == "0.30"
    assert "**Dictamen: FAVORABLE.**" in body["markdown"]

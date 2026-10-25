from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.reports.schemas import CashFlowMetrics, ReportRequest

METRICS = dict(revenue=1200.5, ebitda=300, operating_cash_flow=250, capex=100, cash=80, net_debt=450)


def request(**overrides):
    data = dict(company_name="Cooperativa Andina", period="Q2 2026", metrics=METRICS)
    data.update(overrides)
    return ReportRequest.model_validate(data)


def test_valid_request_and_derived_metrics():
    req = request()
    m = req.metrics
    assert req.language == "es" and req.generation.max_attempts == 3
    assert m.revenue == Decimal("1200.5")  # JSON floats become exact decimals
    assert m.free_cash_flow == Decimal("150")
    assert m.fcf_conversion_pct == Decimal("50")
    assert m.net_leverage == Decimal("1.5")
    dumped = req.model_dump()["metrics"]
    assert dumped["free_cash_flow"] == Decimal("150")  # computed fields are part of the payload


def test_ratios_undefined_without_positive_ebitda():
    m = CashFlowMetrics(**{**METRICS, "ebitda": -10})
    assert m.fcf_conversion_pct is None and m.net_leverage is None


@pytest.mark.parametrize("overrides", [
    {"metrics": {**METRICS, "ebitda": 2000}},             # ebitda > revenue
    {"metrics": {**METRICS, "revenue": 0}},
    {"metrics": {**METRICS, "capex": -1}},
    {"metrics": {**METRICS, "cash": "12.345"}},           # more than 2 decimals
    {"metrics": {**METRICS, "unknown": 1}},               # extra fields rejected
    {"period": "2026-Q2"},
    {"period": "Q5 2026"},
    {"company_name": "Acme <|endoftext|>"},               # no special tokens / markup
    {"company_name": "A"},
    {"language": "fr"},
    {"report_type": "board_minutes"},
    {"generation": {"temperature": 1.5}},
])
def test_rejects_invalid_input(overrides):
    with pytest.raises(ValidationError):
        request(**overrides)


def test_accepts_accented_company_names():
    assert request(company_name="Compañía Ñandú & Asociados, S.A.").company_name.startswith("Compañía")

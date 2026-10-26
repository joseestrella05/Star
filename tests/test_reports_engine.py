from fastapi.testclient import TestClient

from src.api.server import create_app
from src.reports.engine import generate_report
from src.reports.schemas import ModelInfo, ReportRequest
from tests.test_api import MODEL, TOK

INFO = ModelInfo(checkpoint_sha256="abc", params=1)
PAYLOAD = dict(company_name="Atlas Logistics", period="Q2 2026", language="en",
               metrics=dict(revenue=1200, ebitda=300, operating_cash_flow=250, capex=100, cash=80, net_debt=450))


def scripted(*replies):
    """Fake `complete` returning the given continuations in order and recording seeds."""
    calls = []

    def complete(prompt, max_new_tokens, temperature, seed):
        calls.append(seed)
        return replies[len(calls) - 1]
    complete.calls = calls
    return complete


# PAYLOAD metrics: leverage 1.5x and positive FCF, so the rule calls for the "adequate liquidity" closing.
LIQUIDITY = "Liquidity remains adequate to fund the investment plan without additional borrowing."
DELEVERAGING = "The treasury team will prioritize deleveraging before considering share buybacks."


def test_accepts_verified_commentary():
    req = ReportRequest(**PAYLOAD)
    r = generate_report(req, scripted(f" {LIQUIDITY}\n<junk on next line 9>"), INFO)
    assert r.verification.passed and r.verification.attempts == 1
    assert r.verification.rejected_commentary == []
    commentary = r.sections[-1]
    assert commentary.source == "model" and commentary.markdown == LIQUIDITY
    assert r.markdown.startswith("# Atlas Logistics - Cash Flow Analysis, Q2 2026\n\n| Metric | Value (thousands) |")
    assert [s.source for s in r.sections] == ["computed", "computed", "computed", "model"]


def test_retries_then_falls_back_with_seeds():
    req = ReportRequest(**{**PAYLOAD, "generation": {"max_attempts": 2, "seed": 10}})
    fake = scripted(" Revenue grew 45% YoY.", " Net debt fell to 12.0.")
    r = generate_report(req, fake, INFO)
    assert fake.calls == [10, 11]
    assert not r.verification.passed and r.verification.attempts == 2
    assert r.verification.rejected_figures == ["45%", "12.0"]
    assert r.sections[-1].source == "fallback"
    assert "Liquidity remains adequate" in r.markdown


def test_endpoint_with_real_tiny_model():
    with TestClient(create_app(MODEL, TOK, checkpoint_sha256="abc")) as c:
        resp = c.post("/api/v1/reports/generate", json=PAYLOAD)
        body = resp.json()
        assert resp.status_code == 200
        assert body["model"] == {"checkpoint_sha256": "abc", "params": MODEL.num_params(non_embedding=False)}
        assert body["request"]["metrics"]["free_cash_flow"] == "150"
        assert "| Free cash flow (FCF) | 150 |" in body["markdown"]
        assert c.post("/api/v1/reports/generate", json={**PAYLOAD, "period": "2026"}).status_code == 422


def test_rejects_closing_that_contradicts_the_metrics_then_retries():
    req = ReportRequest(**{**PAYLOAD, "generation": {"max_attempts": 3, "seed": 0}})
    r = generate_report(req, scripted(f" {DELEVERAGING}", f" {LIQUIDITY}"), INFO)
    assert r.verification.passed and r.verification.attempts == 2
    assert r.verification.rejected_commentary == [DELEVERAGING]
    assert r.verification.rejected_figures == []  # its figures were fine: it was rejected for contradiction
    assert r.sections[-1].source == "model" and r.sections[-1].markdown == LIQUIDITY


def test_contradicting_closings_fall_back_to_the_rule():
    req = ReportRequest(**{**PAYLOAD, "generation": {"max_attempts": 2}})
    r = generate_report(req, scripted(f" {DELEVERAGING}", " Free cash flow of 150 covers CAPEX."), INFO)
    assert not r.verification.passed and r.verification.attempts == 2
    assert r.verification.rejected_commentary == [DELEVERAGING, "Free cash flow of 150 covers CAPEX."]
    assert r.sections[-1].source == "fallback" and r.sections[-1].markdown == LIQUIDITY

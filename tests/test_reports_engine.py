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


def test_accepts_verified_commentary():
    req = ReportRequest(**PAYLOAD)
    r = generate_report(req, scripted(" Free cash flow of $150.0M covers CAPEX.\n<junk on next line $9M>"), INFO)
    assert r.verification.passed and r.verification.attempts == 1
    commentary = r.sections[-1]
    assert commentary.source == "model" and commentary.markdown == "Free cash flow of $150.0M covers CAPEX."
    assert r.markdown.startswith("# Atlas Logistics - Cash Flow Analysis, Q2 2026\n\n| Metric | Value |")
    assert [s.source for s in r.sections] == ["computed", "computed", "computed", "model"]


def test_retries_then_falls_back_with_seeds():
    req = ReportRequest(**{**PAYLOAD, "generation": {"max_attempts": 2, "seed": 10}})
    fake = scripted(" Revenue grew 45% YoY.", " Net debt fell to $12.0M.")
    r = generate_report(req, fake, INFO)
    assert fake.calls == [10, 11]
    assert not r.verification.passed and r.verification.attempts == 2
    assert r.verification.rejected_figures == ["45%", "$12.0M"]
    assert r.sections[-1].source == "fallback"
    assert "Liquidity remains adequate" in r.markdown


def test_endpoint_with_real_tiny_model():
    with TestClient(create_app(MODEL, TOK, checkpoint_sha256="abc")) as c:
        resp = c.post("/api/v1/reports/generate", json=PAYLOAD)
        body = resp.json()
        assert resp.status_code == 200
        assert body["model"] == {"checkpoint_sha256": "abc", "params": MODEL.num_params(non_embedding=False)}
        assert body["request"]["metrics"]["free_cash_flow"] == "150"
        assert "$150.0M" in body["markdown"]
        assert c.post("/api/v1/reports/generate", json={**PAYLOAD, "period": "2026"}).status_code == 422

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.reports.export import no_fetch, to_html
from tests.test_api import MODEL, TOK

REPORT = "# Atlas - Cash Flow Analysis, Q2 2026\n\n| Metric | Value |\n|---|---:|\n| FCF | $150.0M |\n\nR&D grew."

try:
    import weasyprint  # noqa: F401
    HAS_PDF = True
except OSError:  # Pango not installed (on macOS also set DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib)
    HAS_PDF = False


def client():
    return TestClient(create_app(MODEL, TOK))


def test_html_renders_table_and_escapes_raw_html():
    out = to_html(REPORT + '\n\n<script>alert(1)</script><img src="http://evil/x.png">', footer="id <1>")
    assert "<h1>Atlas - Cash Flow Analysis, Q2 2026</h1>" in out and ">$150.0M</td>" in out
    assert "R&amp;D grew." in out
    assert "<script>" not in out and "<img" not in out
    assert "id &lt;1&gt;" in out


def test_fetcher_blocks_all_urls():
    with pytest.raises(ValueError):
        no_fetch("http://example.com/logo.png")


def test_markdown_export_returns_edited_text():
    with client() as c:
        r = c.post("/api/v1/reports/export", json={"markdown": REPORT, "format": "md"})
    assert r.status_code == 200 and r.text == REPORT
    assert r.headers["content-disposition"] == 'attachment; filename="star-report.md"'
    assert r.headers["content-type"].startswith("text/markdown")


def test_export_validation():
    with client() as c:
        assert c.post("/api/v1/reports/export", json={"markdown": "", "format": "md"}).status_code == 422
        assert c.post("/api/v1/reports/export", json={"markdown": "x", "format": "docx"}).status_code == 422


@pytest.mark.skipif(not HAS_PDF, reason="WeasyPrint system libraries unavailable")
def test_pdf_export():
    with client() as c:
        r = c.post("/api/v1/reports/export",
                   json={"markdown": REPORT + '\n\n![logo](http://example.com/x.png)', "footer": "Report 1"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF") and len(r.content) > 1000

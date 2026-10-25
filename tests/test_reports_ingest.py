import io
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

from src.api.server import create_app
from src.reports.ingest import IngestError, parse_number, parse_upload, template_csv
from tests.test_api import MODEL, TOK

FILLED = {"company_name": "Cooperativa Andina", "revenue": "842.3", "ebitda": "187.6",
          "operating_cash_flow": "151.2", "capex": "64.9", "cash": "95", "net_debt": "310.5"}


def filled_template(values=FILLED, delimiter=","):
    rows = [line.split(",") for line in template_csv().strip().split("\n")]
    for row in rows:
        if row[0] in values:
            row[1] = values[row[0]]
    return "\n".join(delimiter.join(r) for r in rows).encode("utf-8")


@pytest.mark.parametrize("text,expected", [
    ("1,234.5", "1234.5"), ("1.234,5", "1234.5"), ("$ 842.3", "842.3"), ("-12", "-12"),
    ("1,5", "1.5"), ("1,234", "1234"), ("1.234.567,89", "1234567.89"), ("12", "12"),
])
def test_parse_number(text, expected):
    assert parse_number(text) == Decimal(expected)


def test_filled_template_roundtrip():
    result = parse_upload(filled_template())
    assert result["fields"] == {"company_name": "Cooperativa Andina", "period": "Q2 2026", "language": "es"}
    assert result["metrics"].free_cash_flow == Decimal("86.3")


def test_spanish_excel_csv_with_semicolons_and_latin1():
    values = {**FILLED, "company_name": "Compañía Ñandú", "revenue": "1.234,5"}
    data = filled_template(values, delimiter=";").decode("utf-8").encode("latin-1")
    result = parse_upload(data)
    assert result["fields"]["company_name"] == "Compañía Ñandú"
    assert result["metrics"].revenue == Decimal("1234.5")


def test_xlsx_with_numeric_cells():
    wb = Workbook()
    ws = wb.active
    ws.append(["field", "value"])
    for key, value in FILLED.items():
        ws.append([key, value if key == "company_name" else float(value)])
    ws.append(["capex_extra_note", None])  # unknown keys are rejected even when blank
    buf = io.BytesIO()
    wb.save(buf)
    with pytest.raises(IngestError, match="unknown field 'capex_extra_note'"):
        parse_upload(buf.getvalue())
    ws.delete_rows(ws.max_row)
    ws.append(["working_capital_change", 0.1 + 0.2])  # float artifact from a formula -> rounded to cents
    buf = io.BytesIO()
    wb.save(buf)
    m = parse_upload(buf.getvalue())["metrics"]
    assert m.revenue == Decimal("842.3") and m.working_capital_change == Decimal("0.30")


@pytest.mark.parametrize("values,message", [
    ({**FILLED, "revenue": "abc"}, "'revenue' is not a number"),
    ({k: v for k, v in FILLED.items() if k != "cash"}, "cash: Field required"),
    ({**FILLED, "ebitda": "9000"}, "ebitda cannot exceed revenue"),
])
def test_errors_are_reported(values, message):
    with pytest.raises(IngestError, match=message):
        parse_upload(filled_template(values))


def test_rejects_large_and_corrupt_files():
    with pytest.raises(IngestError, match="larger than"):
        parse_upload(b"x" * 1_000_001)
    with pytest.raises(IngestError, match="could not read XLSX"):
        parse_upload(b"PK\x03\x04garbage")


def test_endpoints():
    with TestClient(create_app(MODEL, TOK)) as c:
        t = c.get("/api/v1/reports/template.csv")
        assert t.status_code == 200 and t.text.startswith("field,value,description")
        ok = c.post("/api/v1/reports/ingest", content=filled_template(), headers={"content-type": "text/csv"})
        assert ok.status_code == 200 and ok.json()["metrics"]["free_cash_flow"] == "86.3"
        bad = c.post("/api/v1/reports/ingest", content=b"field,value\nfoo,1\n")
        assert bad.status_code == 422 and bad.json()["detail"] == ["row 2: unknown field 'foo'"]

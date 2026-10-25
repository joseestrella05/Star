from src.reports.formatting import money, num, pct


def test_money_format():
    assert money(1234.56, "en") == "$1,234.6M"
    assert money(-1234.56, "es") == "-$1.234,6 M"


def test_num_and_pct():
    assert num(1234567.891, "es", 2) == "1.234.567,89"
    assert pct(12.34, "en") == "12.3%" and pct(12.34, "es") == "12,3%"

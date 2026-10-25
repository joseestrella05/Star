from scripts.generate_business_corpus import EOT, dcf, generate_corpus


def test_deterministic_and_bilingual():
    docs = generate_corpus(200, seed=7)
    assert docs == generate_corpus(200, seed=7)
    text = "\n".join(docs)
    assert EOT not in text
    for term in ("EBITDA", "CAPEX", "OPEX", "YoY", "ROI", "Estado de Resultados", "Income Statement"):
        assert term in text, term


def test_dcf_flat_case():
    # Zero growth, zero capex, zero tax: EV must equal the perpetuity FCF / WACC.
    _, ev = dcf(100, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.0, 5)
    assert abs(ev - 200) < 1e-9

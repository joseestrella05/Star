from scripts.generate_business_corpus import EOT, generate_corpus
from src.tokenizer.bpe import BPETokenizer

TEXT = f"\n{EOT}\n".join(generate_corpus(300, seed=3))


def trained():
    tok = BPETokenizer()
    tok.train(TEXT, 600)
    return tok


def test_roundtrip_and_specials():
    tok = trained()
    assert tok.vocab_size == 600
    s = f"Los ingresos crecieron 12,3% YoY. {EOT} EBITDA of -$1,234.5M in Q3 2025 — ñandú 🚀"
    ids = tok.encode(s)
    assert tok.decode(ids) == s
    assert ids.count(tok.special_tokens[EOT]) == 1
    assert tok.special_tokens[EOT] not in tok.encode_ordinary(s)


def test_figures_and_terms_are_single_chunks():
    chunks = BPETokenizer().compiled.findall("EBITDA rose 12.3% to -$1,234.5M in Q3; ROIC")
    assert chunks == ["EBITDA", " rose", " 12.3%", " to", " -$1,234.5M", " in", " Q3", ";", " ROIC"]
    tok = trained()
    assert len(tok.encode(" EBITDA")) == 1


def test_save_load(tmp_path):
    tok = trained()
    path = tmp_path / "tok.json"
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    assert tok2.merges == tok.merges and tok2.special_tokens == tok.special_tokens
    assert tok2.encode(TEXT[:5000]) == tok.encode(TEXT[:5000])

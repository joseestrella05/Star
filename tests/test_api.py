import torch
from fastapi.testclient import TestClient

from src.api.server import create_app, load_model
from src.model.config import StarConfig
from src.model.star_gpt import StarGPT
from src.tokenizer.bpe import BPETokenizer
from src.training.trainer import TrainConfig, Trainer

TOK = BPETokenizer()
TOK.train("Revenue grew 12% YoY while EBITDA margin expanded in Q3. " * 20, 300)
MODEL = StarGPT(StarConfig(vocab_size=TOK.vocab_size, block_size=32, n_embd=32, n_head=4, n_layer=2)).eval()


def client():
    return TestClient(create_app(MODEL, TOK))


def test_health():
    with client() as c:
        body = c.get("/health").json()
    assert body["status"] == "ok" and body["config"]["n_embd"] == 32


def test_generate():
    with client() as c:
        r = c.post("/generate", json={"prompt": "EBITDA margin", "max_new_tokens": 8, "temperature": 0})
    body = r.json()
    assert r.status_code == 200
    assert body["prompt_tokens"] == len(TOK.encode("EBITDA margin"))
    assert 0 <= body["completion_tokens"] <= 8


def test_validation():
    with client() as c:
        assert c.post("/generate", json={"prompt": "x", "temperature": 5}).status_code == 422
        assert c.post("/generate", json={"prompt": ""}).status_code == 422
        assert c.post("/generate", json={"prompt": "x", "max_new_tokens": 10_000}).status_code == 422
        assert c.post("/generate", json={"prompt": "x", "max_new_tokens": 5, "min_new_tokens": 6}).status_code == 422


def test_min_new_tokens_forces_full_length():
    with client() as c:
        body = c.post("/generate", json={"prompt": "string", "max_new_tokens": 20, "min_new_tokens": 20}).json()
    assert body["completion_tokens"] == 20


def test_load_model_roundtrip(tmp_path):
    trainer = Trainer(MODEL, None, None, TrainConfig(out_dir=str(tmp_path)), device="cpu",
                      extra_state={"tokenizer": "tok.json"})
    path = trainer.save(0, 1.0)
    model, tok_path = load_model(path, "cpu")
    assert tok_path == "tok.json"
    x = torch.randint(0, TOK.vocab_size, (1, 8))
    assert torch.equal(model(x)[0], MODEL(x)[0])

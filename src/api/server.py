"""FastAPI inference server for StarGPT.

Run: STAR_CHECKPOINT=checkpoints/best.pt uvicorn src.api.server:app --port 8000
"""
import hashlib
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from src.api.routers import reports
from src.model.config import StarConfig
from src.model.star_gpt import StarGPT
from src.tokenizer.bpe import EOT, BPETokenizer
from src.training.trainer import get_device


class GenerateRequest(BaseModel):
    # In-domain example: the Swagger default "string" is out of domain and makes the model emit EOS at once.
    model_config = {"json_schema_extra": {"examples": [{
        "prompt": "Atlas Logistics - Cash Flow Analysis, Q2 2025", "max_new_tokens": 150, "min_new_tokens": 0,
        "temperature": 0.8, "top_k": 50, "top_p": 0.95,
    }]}}

    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(128, ge=1, le=512)
    min_new_tokens: int = Field(0, ge=0, description="EOS is suppressed until this many tokens are generated")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="0 = greedy decoding")
    top_k: Optional[int] = Field(50, ge=1)
    top_p: Optional[float] = Field(0.95, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def check_lengths(self):
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens must be <= max_new_tokens")
        return self


class GenerateResponse(BaseModel):
    completion: str
    prompt_tokens: int
    completion_tokens: int


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model = StarGPT(StarConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt.get("tokenizer")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def create_app(model=None, tokenizer=None, device="cpu", checkpoint_sha256="unknown"):
    """Build the app; model/tokenizer are loaded from env vars at startup when not injected."""
    state = {"model": model, "tokenizer": tokenizer, "device": device, "checkpoint_sha256": checkpoint_sha256}
    lock = threading.Lock()  # one generation at a time on the shared device

    def complete(prompt, max_new_tokens, temperature, seed=None, top_k=None, top_p=None, min_new_tokens=0):
        """Continue `prompt`; returns (text, prompt_tokens, completion_tokens), cut at EOS."""
        model, tok = state["model"], state["tokenizer"]
        eos = tok.special_tokens.get(EOT)
        ids = tok.encode(prompt, allowed_special=False)  # user text cannot inject special tokens
        idx = torch.tensor([ids], dtype=torch.long, device=state["device"])
        with lock:
            if seed is not None:
                torch.manual_seed(seed)
            out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                                 eos_id=eos, min_new_tokens=min_new_tokens)
        new = out[0, len(ids):].tolist()
        if eos in new:
            new = new[: new.index(eos)]
        return tok.decode(new), len(ids), len(new)

    state["complete"] = lambda prompt, max_new_tokens, temperature, seed: complete(
        prompt, max_new_tokens, temperature, seed)[0]

    @asynccontextmanager
    async def lifespan(_app):
        if state["model"] is None:
            checkpoint = os.environ.get("STAR_CHECKPOINT", "checkpoints/best.pt")
            state["device"] = os.environ.get("STAR_DEVICE") or get_device()
            state["model"], tok_path = load_model(checkpoint, state["device"])
            state["checkpoint_sha256"] = sha256_file(checkpoint)
            state["tokenizer"] = BPETokenizer.load(
                os.environ.get("STAR_TOKENIZER") or tok_path or "checkpoints/tokenizer.json")
        yield

    # Swagger UI and ReDoc load their assets from a CDN; STAR_DOCS=0 turns them off for offline deployments.
    docs = os.environ.get("STAR_DOCS", "1") != "0"
    app = FastAPI(title="Star API", version="0.1.0", lifespan=lifespan,
                  docs_url="/docs" if docs else None, redoc_url="/redoc" if docs else None)
    app.state.star = state
    app.include_router(reports.router)

    @app.get("/health")
    def health():
        m = state["model"]
        return {"status": "ok", "device": state["device"], "params": m.num_params(non_embedding=False),
                "config": vars(m.config), "checkpoint_sha256": state["checkpoint_sha256"]}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):  # sync def: FastAPI runs it in a worker thread
        text, n_prompt, n_new = complete(req.prompt, req.max_new_tokens, req.temperature, top_k=req.top_k,
                                         top_p=req.top_p, min_new_tokens=req.min_new_tokens)
        return GenerateResponse(completion=text, prompt_tokens=n_prompt, completion_tokens=n_new)

    # Built dashboard (web/dist), mounted last so API routes take precedence.
    web_dir = Path(os.environ.get("STAR_WEB_DIR", "web/dist"))
    if (web_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app


app = create_app()

"""FastAPI inference server for StarGPT.

Run: STAR_CHECKPOINT=checkpoints/best.pt uvicorn src.api.server:app --port 8000
"""
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

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


def create_app(model=None, tokenizer=None, device="cpu"):
    """Build the app; model/tokenizer are loaded from env vars at startup when not injected."""
    state = {"model": model, "tokenizer": tokenizer, "device": device}
    lock = threading.Lock()  # one generation at a time on the shared device

    @asynccontextmanager
    async def lifespan(_app):
        if state["model"] is None:
            state["device"] = os.environ.get("STAR_DEVICE") or get_device()
            state["model"], tok_path = load_model(os.environ.get("STAR_CHECKPOINT", "checkpoints/best.pt"),
                                                  state["device"])
            state["tokenizer"] = BPETokenizer.load(
                os.environ.get("STAR_TOKENIZER") or tok_path or "checkpoints/tokenizer.json")
        yield

    app = FastAPI(title="Star API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        m = state["model"]
        return {"status": "ok", "device": state["device"], "params": m.num_params(non_embedding=False),
                "config": vars(m.config)}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):  # sync def: FastAPI runs it in a worker thread
        model, tok = state["model"], state["tokenizer"]
        eos = tok.special_tokens.get(EOT)
        ids = tok.encode(req.prompt, allowed_special=False)  # user text cannot inject special tokens
        idx = torch.tensor([ids], dtype=torch.long, device=state["device"])
        with lock:
            out = model.generate(idx, req.max_new_tokens, temperature=req.temperature,
                                 top_k=req.top_k, top_p=req.top_p, eos_id=eos, min_new_tokens=req.min_new_tokens)
        new = out[0, len(ids):].tolist()
        if eos in new:
            new = new[: new.index(eos)]
        return GenerateResponse(completion=tok.decode(new), prompt_tokens=len(ids), completion_tokens=len(new))

    return app


app = create_app()

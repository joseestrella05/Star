# Star — Business-Domain Small Language Model

Decoder-only Transformer (GPT-style) SLM for business, corporate finance and executive analysis.

## Workflow rules (mandatory)
- No AI attribution anywhere: no `Co-authored-by`, no Anthropic/AI mentions in code, comments, docs or commit messages.
- Never run `git commit` or `git push`. After each phase: `git add` the phase files only, then hand over the exact
  `git commit -m "<type>(<scope>): <description>"` command (Conventional Commits, English).
- Do not start the next phase until the user confirms the commit.
- Minimal console output: no greetings, no theory, no re-pasting code already written to files.

## Tech spec
- Device: auto-detect `mps` > `cuda` > `cpu`.
- Tokenizer: byte-level BPE from scratch (pure Python), save/load, regex preserving financial figures,
  percentages and terms (`EBITDA`, `ROI`, `Q1-Q4`, `YoY`, `CAPEX`, `OPEX`).
- Model `StarGPT` (PyTorch): learned token + positional embeddings; Pre-LN blocks; causal MHSA via
  `scaled_dot_product_attention`; GELU FFN; residuals; `lm_head` weight-tied to token embeddings.
- Base config: `vocab_size=4096, n_embd=256, n_head=8, n_layer=6, block_size=256, dropout=0.1` (~10-15M params).
- Training: AdamW, cosine LR with warmup, grad clip 1.0, train/val cross-entropy, `.pt` checkpoints.
- Inference: autoregressive generation with `temperature`, `top_k`, `top_p`; FastAPI REST server.

## Layout
`src/tokenizer`, `src/model`, `src/training`, `src/api`, `scripts`, `data`, `tests`

## Phase status
- [x] 1 `chore(setup)` — CLAUDE.md, .gitignore, requirements.txt, folder structure
- [ ] 2 `feat(data)` — `scripts/generate_business_corpus.py` -> `data/business_corpus.txt` (ES/EN)
- [ ] 3 `feat(tokenizer)` — `src/tokenizer/bpe.py` (train/encode/decode/save/load), `scripts/train_tokenizer.py`
- [ ] 4 `feat(attention)` — `src/model/attention.py` (`CausalSelfAttention`, embeddings)
- [ ] 5 `feat(model)` — `src/model/star_gpt.py` (`FeedForward`, `TransformerBlock`, `StarGPT`, `generate()`)
- [ ] 6 `feat(training)` — `src/training/dataset.py`, `src/training/trainer.py`, `scripts/train.py`
- [ ] 7 `feat(api)` + `docs(readme)` — `src/api/server.py`, `README.md`

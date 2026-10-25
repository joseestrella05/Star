# Star

**Star** is a small decoder-only Transformer language model (GPT architecture) built from scratch in PyTorch and
specialized in business, corporate finance and executive analysis, in English and Spanish. Every component, from the
byte-level BPE tokenizer to the training loop and the inference API, is implemented in this repository with no
dependency on pretrained weights or third-party tokenizers.

```
corpus generator ──> BPE tokenizer ──> token stream ──> StarGPT training ──> checkpoint ──> FastAPI /generate
```

## Highlights

- **Byte-level BPE from scratch** (pure Python) with domain-aware pre-tokenization: figures like `-$1,234.5M`,
  `12,3%`, `7.9x` and terms like `EBITDA`, `CAPEX`, `OPEX`, `ROI`, `YoY`, `Q1-Q4` are never split across chunks.
  Each corporate term ends up as a single token. Trains on 6.8M characters in ~4 s through incremental pair counting.
- **GPT-style model**: learned token and positional embeddings, Pre-LayerNorm blocks, multi-head causal
  self-attention on `scaled_dot_product_attention` (Flash / memory-efficient kernels when available), GELU MLP,
  residual connections and weight tying between `lm_head` and the token embedding.
- **Training**: AdamW with decoupled weight decay on matrices only, linear warmup + cosine decay, gradient clipping,
  train/val evaluation and best-checkpoint saving. Auto-selects `mps`, `cuda` or `cpu`.
- **Serving**: FastAPI with validated requests, temperature / top-k / nucleus (top-p) sampling and EOS stopping.
- **Tested**: 21 unit tests covering tokenizer round-trips, attention causality, flash vs. explicit-mask
  equivalence, sampling filters, LR schedule, training convergence and the API.

## Project layout

```
scripts/
  generate_business_corpus.py   synthetic bilingual corpus -> data/business_corpus.txt
  train_tokenizer.py            BPE training -> checkpoints/tokenizer.json
  train.py                      model pre-training -> checkpoints/best.pt
src/
  tokenizer/bpe.py              BPETokenizer: train / encode / decode / save / load
  model/config.py               StarConfig
  model/attention.py            Embeddings, CausalSelfAttention
  model/star_gpt.py             FeedForward, TransformerBlock, StarGPT, sampling filters
  training/dataset.py           token cache and shifted (X, Y) windows
  training/trainer.py           Trainer, LR schedule, optimizer, device detection
  api/server.py                 FastAPI app
  api/routers/reports.py        /api/v1/reports endpoints
  reports/formatting.py         financial number formatting shared by corpus and report prompts
  reports/schemas.py            typed report request/response contracts
  reports/templates.py          metrics -> canonical prompt, table and fallback commentary
  reports/verify.py             figure extraction and verification
  reports/engine.py             prompt -> generate -> verify -> Markdown report
  reports/export.py             Markdown -> HTML -> PDF (offline)
tests/                          pytest suite
```

## Quickstart

Requires Python 3.9+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_business_corpus.py          # 10k docs, ~6.8M chars
python scripts/train_tokenizer.py                   # vocab 4096
python scripts/train.py --max-iters 5000            # trains on mps / cuda / cpu
uvicorn src.api.server:app --port 8000              # serves checkpoints/best.pt

pytest -q
```

```bash
curl -s localhost:8000/generate -H 'content-type: application/json' -d '{
  "prompt": "Atlas Logistics - Cash Flow Analysis, Q2 2025",
  "max_new_tokens": 120, "temperature": 0.7, "top_k": 50, "top_p": 0.95
}'
```

| Endpoint | Description |
|---|---|
| `GET /health` | device, parameter count, model config and checkpoint SHA-256 |
| `POST /api/v1/reports/generate` | typed cash-flow metrics (USD millions) -> Markdown report: computed table and analysis, plus model commentary kept only if every figure in it is verified (deterministic fallback otherwise) |
| `POST /api/v1/reports/export` | `{markdown, format: "md" \| "pdf", footer}` -> file download; takes the user-edited text, escapes raw HTML and renders the PDF with all network/file fetching disabled |

PDF export uses WeasyPrint, which needs the system Pango library (`brew install pango` on macOS,
`apt install libpango-1.0-0 libpangoft2-1.0-0` on Debian/Ubuntu). On macOS run the server with
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`. Without Pango, PDF export returns 503 and everything else works.
| `POST /generate` | `prompt` (1-4000 chars), `max_new_tokens` (1-512), `min_new_tokens` (EOS suppressed until reached; set equal to `max_new_tokens` to force full length), `temperature` (0-2, 0 = greedy), `top_k` (>=1), `top_p` (0-1] |

Generation stops early only when the model samples `<|endoftext|>`. The model was trained on whole documents, so
an out-of-domain prompt, or one ending in a trailing space, often makes it predict end-of-document within a few
tokens. Use in-domain prompts or `min_new_tokens`.

Server settings come from environment variables: `STAR_CHECKPOINT` (default `checkpoints/best.pt`),
`STAR_TOKENIZER` (defaults to the path recorded in the checkpoint) and `STAR_DEVICE`.

## Data

`generate_business_corpus.py` produces five document types: income statements, cash-flow analyses, board minutes,
KPI dashboards and multi-year DCF projections. Each language uses its own number formatting (`$1,234.5M` vs.
`$1.234,5 M`). Every document is derived from one internally consistent set of financials, so gross profit, EBITDA,
FCF, margins, leverage and enterprise value agree with each other inside the text. The corpus is deterministic for a
given `--seed`. Documents are separated by the special token `<|endoftext|>`.

## Tokenizer: byte-level BPE

Text is first split into chunks by a regex. Merges never cross chunk boundaries, which keeps a financial figure
or a term like `EBITDA` from merging with neighbouring words. Each chunk starts as its UTF-8 bytes, so the base
vocabulary is the 256 byte values and any input is encodable, accents and emoji included.

Training repeats, for $k = 1 \dots V - 256 - |S|$:

$$
(a, b)^{*} = \arg\max_{(a,b)} \sum_{w} f(w)\, \mathrm{count}_{w}(a, b), \qquad \text{vocab}[255 + k] = \text{vocab}[a^{*}] \,\Vert\, \text{vocab}[b^{*}]
$$

where $w$ ranges over unique chunks with corpus frequency $f(w)$ and $S$ is the set of special tokens. Two
optimizations make pure Python practical:

1. **Chunk deduplication**: pairs are counted over unique chunks weighted by frequency, not over the raw text.
2. **Incremental updates**: an index from pair to the chunks containing it means each merge only rewrites and
   recounts the affected chunks instead of rescanning everything.

Encoding applies merges greedily by rank (earliest learned first), reproducing training order. Results are cached
per chunk. Special tokens are split out before the regex. The API encodes user prompts with special tokens
disabled, so a prompt cannot inject `<|endoftext|>`.

Result on the default corpus: **4.18 bytes/token** on held-out text.

## Model: StarGPT

For an input sequence $x_{1..T}$:

$$
h_0 = E[x] + P[1..T], \qquad
h'_l = h_{l-1} + \mathrm{MHA}(\mathrm{LN}(h_{l-1})), \qquad
h_l = h'_l + \mathrm{FFN}(\mathrm{LN}(h'_l))
$$

$$
\mathrm{logits} = \mathrm{LN}(h_L)\, E^{\top}
$$

**Causal self-attention.** Per head, with $d_k = n_{embd} / n_{head}$:

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right) V,
\qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

Q, K and V come from one fused linear projection. Head outputs are concatenated and projected back to $n_{embd}$.
The default path calls `F.scaled_dot_product_attention(is_causal=True)`. With `flash=False` the model uses an
explicit lower-triangular mask instead, and the tests check that both paths match to 1e-5.

**FFN.** $\mathrm{FFN}(x) = W_2\,\mathrm{GELU}(W_1 x)$ with a $4\times$ hidden expansion.

**Pre-LN** puts LayerNorm before each sublayer, which keeps the residual stream an identity path and makes
training stable without careful warmup tuning. A final LayerNorm precedes the output head.

**Weight tying.** `lm_head.weight` is the token embedding matrix $E$, saving $V \cdot n_{embd}$ parameters and
coupling input and output representations.

**Initialization.** All weights are $\mathcal{N}(0, 0.02)$ and biases are zero. Residual output projections use
$\mathcal{N}(0, 0.02/\sqrt{2L})$ so the variance of the residual stream does not grow with depth.

### Configuration

| | default |
|---|---|
| `vocab_size` | 4096 |
| `block_size` | 256 |
| `n_embd` | 256 |
| `n_head` | 8 (head dim 32) |
| `n_layer` | 6 |
| `dropout` | 0.1 |
| **parameters** | **5.85M** |

Per layer the model holds about $12\, n_{embd}^2$ weights (attention $4 n_{embd}^2$, FFN $8 n_{embd}^2$), so the
total is roughly $12 L\, n_{embd}^2 + V n_{embd} + T n_{embd}$. To scale up, pass `--n-embd 384` (~12.3M) or
`--n-layer 12` (~10.6M) to `scripts/train.py`.

## Training

- **Objective**: next-token cross-entropy $\mathcal{L} = -\frac{1}{T}\sum_t \log p_\theta(x_{t+1} \mid x_{\le t})$
  over windows $X = t[i:i+T]$, $Y = t[i+1:i+T+1]$ drawn from the token stream.
- **Split**: the last 10% of the stream is validation, kept contiguous so no window overlaps the training data.
- **Optimizer**: AdamW, $\beta = (0.9, 0.95)$, weight decay 0.1 on 2-D parameters only (no decay on biases or
  LayerNorm gains).
- **Schedule**: linear warmup to the peak LR, then cosine decay to 10% of it:
  $\eta_t = \eta_{min} + \tfrac12(1 + \cos(\pi\,\tfrac{t - t_w}{t_{max} - t_w}))(\eta_{max} - \eta_{min})$.
- **Stability**: gradient-norm clipping at 1.0.
- **Checkpoints**: `best.pt` stores weights, model and training config, iteration, validation loss and the tokenizer
  path, so a checkpoint is enough to rebuild the model for serving.

Smoke run on Apple Silicon (`mps`), default config, batch 32, 200 iterations: validation loss went from 8.39
($\approx \ln 4096$, uniform) to 1.52.

```bash
python scripts/train.py --help    # all hyperparameters are exposed as flags
```

## Inference

`StarGPT.generate` samples autoregressively, cropping the context to the last `block_size` tokens:

1. Logits for the last position are divided by `temperature` (`0` switches to greedy argmax).
2. **Top-k** keeps the $k$ highest logits.
3. **Top-p** keeps the smallest set of tokens, by descending probability, whose cumulative mass reaches $p$. The
   most likely token is always kept.
4. The next token is sampled from the renormalized distribution. Generation stops at `<|endoftext|>`.

## Limitations

- The corpus is synthetic and template-based. The model learns the structure and vocabulary of financial
  documents, not real-world facts, and generated figures are not arithmetically consistent. Low validation loss
  reflects the corpus's regularity, not general language ability.
- No mixed-precision training, gradient accumulation or resume-from-checkpoint yet.
- Absolute positional embeddings limit context to `block_size` tokens.

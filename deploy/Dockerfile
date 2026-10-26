# Build context: repository root.  docker compose -f deploy/docker-compose.yml build

# --- 1. Dashboard: compiled to static files; Node is not shipped in the runtime image.
FROM node:22.12.0-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# --- 2. Runtime: Python + CPU-only PyTorch, model and dashboard baked in.
FROM python:3.10.15-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Pango for WeasyPrint PDF rendering; DejaVu so PDFs render without downloading fonts.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpango-1.0-0 libpangoft2-1.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU wheel first so requirements.txt does not pull ~2 GB of CUDA libraries. Its dependencies come
# from PyPI: the PyTorch index ships typing-extensions with metadata pip rejects.
RUN pip install --upgrade pip \
    && pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    && pip install -r requirements.txt

COPY src/ src/
COPY --from=web /web/dist web/dist
ARG CHECKPOINT=checkpoints/star-v2.pt
COPY ${CHECKPOINT} model/star.pt
COPY checkpoints/tokenizer.json model/tokenizer.json

ENV STAR_CHECKPOINT=/app/model/star.pt \
    STAR_TOKENIZER=/app/model/tokenizer.json \
    STAR_WEB_DIR=/app/web/dist \
    STAR_DEVICE=cpu \
    STAR_DOCS=0

USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]

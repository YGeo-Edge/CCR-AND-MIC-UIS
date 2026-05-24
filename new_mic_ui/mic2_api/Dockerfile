# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="MIC2 VLM Classifier API"
LABEL org.opencontainers.image.description="Malicious ad detection — InternVL2.5-1B + LoRA"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

# Runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/         ./app/
COPY model.py     ./model.py

# Model weights are mounted at runtime via volume (not baked in — too large)
# Expected path: /app/weights/mic2_internvl_v1/
#   Contains: adapter_config.json, adapter_model.safetensors,
#             classifier_head.pt, model_meta.json

# HuggingFace cache (base model downloaded on first start unless pre-mounted)
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

# App config (override via docker run -e or docker-compose environment)
ENV ADAPTER_PATH=/app/weights/mic2_internvl_v1
ENV BATCH_SIZE=8
ENV BATCH_TIMEOUT_MS=50
ENV JOB_TTL_S=300
ENV MAX_UPLOAD_MB=10
ENV WORKERS=1

EXPOSE 8000

# Non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/weights /app/hf_cache && \
    chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]

# Production image for the loan-default-risk API.
#
# Two things matter beyond the obvious build: the runtime user must own the
# directories that host-provided volumes mount over, and the model must be
# injected rather than baked in, so an image built today can serve a model
# trained tomorrow.

FROM python:3.11-slim AS base

# libgomp1 is required by LightGBM's OpenMP runtime; it is the only system
# package needed, and apt lists are removed in the same layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so the layer is cached until pyproject.toml or
# the lock file changes. The project itself is installed, not copied, so
# src.api.model and src.loan_default_risk.config resolve as declared packages.
COPY pyproject.toml README.md SECURITY.md ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

COPY src ./src
COPY configs ./configs

# Artifacts are mounted at run time; create a writable, owned directory now so
# a volume mount later does not shadow a root-owned path. mlruns is the MLflow
# file store that docker-compose mounts.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts /app/mlruns \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LDR_LOG_LEVEL=INFO \
    LDR_ARTIFACTS_DIR=/app/artifacts \
    LDR_MODEL_PATH=/app/artifacts/model.joblib

EXPOSE 8000

# Healthcheck hits /health, which never touches the model, so a slow-starting
# model does not cause a flapping restart loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

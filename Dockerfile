FROM python:3.11-slim as base
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
# deps from pyproject (no hardcode) - install minimal for API
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir fastapi uvicorn scikit-learn lightgbm pandas pyarrow pydantic mlflow shap
COPY src ./src
COPY configs ./configs
RUN useradd -m appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn","src.api.main:app","--host","0.0.0.0","--port","8000"]

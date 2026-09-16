FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir pandas scikit-learn lightgbm fastapi uvicorn mlflow shap
COPY . .
EXPOSE 8000
CMD ["uvicorn","src.api.main:app","--host","0.0.0.0","--port","8000"]
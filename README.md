# loan-default-risk

Batch tabular risk system predicts PD for loan applicants, profit-based approve/reject at thr 0.05, adverse-action SHAP, A/B.

## Problem & Scale

1.34M LendingClub 2007-18, 151 cols -> 24 origination, temporal train 451k 17% / val 668k 21.5% / test 225k 21.3%, leakage audit docs/leakage_audit.md.

## Architecture

data/raw -> src/data/make_dataset.py -> train/val/test.parquet -> src/features/build_features.py -> src/models/train.py (LGBM isotonic Brier 0.203->0.155) -> src/policy/profit.py thr 0.05 -> src/api/main.py /predict /explain /experiment/assign -> tests + Docker

## Metrics

| Split | ROC   | Gini  | PR-AUC | Brier | KS    | Recall@5% |
| ----- | ----- | ----- | ------ | ----- | ----- | --------- |
| val   | 0.722 | 0.444 | 0.413  | 0.155 | 0.323 | 0.198     |
| test  | 0.703 | 0.407 | 0.371  | 0.156 | 0.298 | 0.156     |

Fairness purpose debt_consolidation 0.381 > credit_card 0.325

## How to Run

pip install -r requirements.txt
python src/data/make_dataset.py
python -m src.models.train
uvicorn src.api.main:app --reload
docker-compose up

## Cost/Latency

Train 5-fold isotonic 3 min on 451k, API p50 126ms p95 172ms mean 129ms \(bench.py\), Brier 0.155, profit +7.4M at 0.05.

## Limitations

No Home Credit data, LGD 0.6 fixed, no real Postgres/Redis, drift -0.04 PR val->test needs quarterly retrain.

## Resume Bullet
Loan Default Risk API - XGBoost isotonic 1.34M PR 0.413 Brier 0.155 profit +7.4M thr 0.05 SHAP FastAPI Docker MLflow [Demo] [GitHub]

## Roadmap

M6 fairness age/gender, Evidently drift, Streamlit dashboard, MLflow registry.


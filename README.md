# loan-default-risk — PD + Profit + SHAP + A/B

[![CI](https://github.com/WHITEJACK5/loan-default-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/WHITEJACK5/loan-default-risk/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](pyproject.toml)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Batch tabular risk system that predicts probability of default (PD) for LendingClub loans and converts it into a profit-based approve/reject policy with adverse-action explanations and A/B experimentation. Built with temporal splits, no accuracy-only, calibrated, Docker one-command.

**Live Demo:** [https://whitejack5-loan-default-risk.hf.space](https://whitejack5-loan-default-risk.hf.space) — Gradio ZeroGPU · **GitHub:** [WHITEJACK5/loan-default-risk](https://github.com/WHITEJACK5/loan-default-risk) · **Tag:** `v1.3.0`

## Problem & Scale

2.26M raw LendingClub 2007-18Q4 → 1.34M after `Fully Paid/Charged Off` filter (drop `Current/Late`). 151 cols → 24 origination-time cols via `docs/leakage_audit.md` (drop `out_prncp, total_pymnt, recoveries, last_pymnt, hardship_20, settlement_7`). Temporal split `train ≤2014 451,059 17.0%` / `val 2015-16 668,640 21.5%` / `test ≥2017 225,611 21.3%` — no random split. Validation `bad_loan 0, bad_target 0` (`src/data/validate.py`).

## Architecture

![Architecture](docs/architecture.png)
`data/raw/392MB` → `src/data/make_dataset.py` (`%b-%Y`, leakage-safe, parquet) → `src/features/build_features.py` (12 num median+scale, 8 cat most_frequent+OneHot) → `src/models/train.py` (Logistic `ROC 0.719 PR 0.408` → LightGBM `0.722/0.413` `class_weight=balanced`) → `src/models/calibrate.py` (isotonic cv5 `Brier 0.204→0.155` `ROC 0.722`) → `src/policy/profit.py` (`LGD 0.6 EAD loan_amnt, thr 0.05 profit +7.4M approve 7.9% default 3.5%`) → `src/api/main.py` (`/predict /explain /experiment/assign`, `src/models/explain_shap.py` top3) → `src/monitoring/drift.py` (PSI + Evidently) → `Docker + MLflow`

## Metrics

| Split        | n       | dr    | ROC   | Gini  | PR-AUC | Brier | KS    | Recall@5% |
| ------------ | ------- | ----- | ----- | ----- | ------ | ----- | ----- | --------- |
| val 2015-16  | 668,640 | 0.215 | 0.722 | 0.444 | 0.413  | 0.155 | 0.323 | 0.198     |
| test 2017-18 | 225,611 | 0.213 | 0.703 | 0.407 | 0.371  | 0.156 | 0.298 | 0.156     |

Fairness `purpose`: `debt_consolidation 0.381 (123k), credit_card 0.325 (44k), home_improvement 0.327 (18k)` — drift `PR -0.042 Gini -0.037` `val→test` → quarterly retrain. See `docs/metrics.md` and `docs/drift_report.html`.

## Results

![Calibration](docs/calibration_curve.png)
_Isotonic Brier 0.204→0.155 ROC 0.722_

![Profit](docs/profit_curve.png)
_Best thr 0.05 profit +7.4M_

![SHAP](docs/shap_summary.png)
_Top drivers int_rate, dti, grade_

## How to Run

```bash
# 1. env
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r pyproject.toml  # or pip install -e .
# 2. data
# download https://www.kaggle.com/datasets/wordsforthewise/lending-club -> data/raw/accepted_2007_to_2018Q4.csv.gz
python src/data/make_dataset.py
python src/data/validate.py
# 3. train + calibrate + profit
python -m src.models.train        # Logistic 0.719/0.408 LightGBM 0.722/0.413
python -m src.models.calibrate    # Brier 0.155 docs/calibration_curve.png
python -m src.policy.profit       # thr 0.05 docs/profit_curve.csv
# 4. serve
uvicorn src.api.main:app --reload --port 8000  # http://127.0.0.1:8000/docs
# or docker
docker build -t loan-risk . && docker run -p 8000:8000 loan-risk
docker-compose up  # api + volumes
# 5. test
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -v  # 3 passed
python scripts/bench.py  # p50 126ms p95 172ms
Cost / Latency
Train LightGBM isotonic cv5 500 trees 3 min on 451k 16GB RAM no GPU. API p50 126ms p95 172ms mean 129ms (scripts/bench.py 50 calls) on FastAPI 512MB Docker, no GPU $0/hr local. HF Gradio ZeroGPU free drift HTML. Brier 0.155 profit +7.4M at thr 0.05.
Limitations
- No Home Credit multi-table joins, single LendingClub only
- LGD 0.6 fixed, op_cost $500 heuristic, not per-grade LGD
- no real Postgres/Redis — assign.py uses md5 hash, deps.py placeholder @lru_cache
- drift input PSI ok but PR -0.04 model drift needs quarterly retrain, Evidently legacy on 0.7.23
- dashboard/app.py Streamlit minimal, not React
Roadmap
- Postgres loan_store + Redis buckets
- Streamlit/Dash profit curve, SHAP force, cohort slicing fully
- MLflow registry + Evidently auto-retrain on PSI >0.2
- Fairness gender/age band + CUPED
- P2 Fraud 1:500 PR-AUC latency 50ms Kafka Redis next
Structure
Project Structure — README, pyproject, Dockerfile, docker-compose, .github/workflows/ci.yml, configs/config.yaml, src/data/, features/, models/, policy/, experiments/, api/, monitoring/, tests/, notebooks/, scripts/, docs/architecture.png, model_card.md, dashboard/app.py
License
MIT
```


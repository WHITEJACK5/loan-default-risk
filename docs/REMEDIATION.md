# Remediation — D1-D32 Verification

All 32 CONFIRMED with cmd excerpts per ramayan Phase 0.

| D | Status | Evidence |
|---|--------|----------|
| D1 | CONFIRMED | `.github/workflows/ci.yml: ruff check . \|\| true` x5, `cov 80` with `\|\| true`, no deps install |
| D2 | CONFIRMED | `notebooks/01_eda.ipynb 553 {"cells":[]}` `print('val ROC 0.722')` hardcoded, `scripts/gen_notebooks.py` exists |
| D3 | CONFIRMED | `src/api/main.py @on_event startup cal.fit` + `data/processed/*.parquet` gitignored `*.parquet` → `docker run` FileNotFound; `Dockerfile FROM python:3.11-slim` no `libgomp1`, root, no HEALTHCHECK |
| D4 | CONFIRMED | `Get-ChildItem artifacts/model.joblib` not found, `grep joblib src/` 0, `api/main.py` retrains 3 min per start, `pytest` retrains 6x |
| D5 | CONFIRMED | `space/` not in repo vs `huggingface.co/spaces/WHITEJACK5` has `app.py @spaces.GPU`; `pyproject 0.1.0` vs `README v1.3.0` vs `app.py v1.1.0` vs tags `v1.0.0-1.3.0` |
| D6 | CONFIRMED | `pyproject.toml` no `[build-system]`, missing `uvicorn, pyarrow, scipy, matplotlib, evidently, streamlit, gradio, pytest, httpx`, `pandera` unused, `pip install -r pyproject.toml` invalid, `Dockerfile` hardcoded unpinned |
| D7 | CONFIRMED | `Test-Path LICENSE` false + MIT badge, `Test-Path AGENTS.md` false + `git grep gita` 27 hits, `README Resume Bullet`, `3802128`, `credit_card 0.325 > 0.327` false, `XGBoost` vs `LGBMClassifier` |
| D8 | CONFIRMED | `docs/metrics.md val 0.413 vs test 0.371` but `README headline 0.413` no split, `honest 0.722/0.371` mix, `Brier 0.203 vs 0.204`, `Recall@5%` is `Recall@FPR5%` |
| D9 | CONFIRMED | `calibration_curve.png` isotonic above diagonal `pred 0.34→true 0.48` (train 17% vs val 21.5%), `class_weight balanced` → `Brier 0.204→0.155` undo, naive Brier `0.168` at 21.3% |
| D10 | CONFIRMED | `make_dataset.py` drops `Current` right-censored, `docs/censoring.md` missing, `17%→21.5% drift` never measured as censoring |
| D11 | CONFIRMED | No baselines `grade/int_rate`, `int_rate/grade/sub_grade/installment` circular, `README top drivers int_rate,dti,grade` vs `shap int_rate,term_36,annual_inc,dti` |
| D12 | CONFIRMED | `profit.py profit_per=(1-dr)*rate*EAD - dr*LGD*EAD -500` `LGD 0.6*full principal` `mean*mean` not sum, ignores `term`, `profit_curve.csv approve-all -623M` |
| D13 | CONFIRMED | `np.arange(0.05,0.96,0.05)` best at boundary `0.05`, `thr` chosen on `val` never test, no baselines, USD unlabeled |
| D14 | CONFIRMED | `evaluate.py fairness PR-AUC by purpose` not comparable, `purpose` not protected, `addr_state` proxy used, no `protected absent` doc |
| D15 | CONFIRMED | `api/main.py /explain np.argsort(-abs)` can list helpful, explains raw 1st fold not calibrated, rebuilds `TreeExplainer` per request, one-hot names |
| D16 | CONFIRMED | `explain_shap.py proba[:200] idx0 but single = X_trans[0:1]` row 0 train, `X_trans[:200]` oldest rows |
| D17 | CONFIRMED | `schemas.py` all fields `=`, missing `ge/le`, `grade` free text, `earliest_cr_line` ignored, missing `dti` defaults to `18` not 422 |
| D18 | CONFIRMED | `config.yaml threshold 0.05 LGD` never read (`grep config.yaml src/` 0), `THRESHOLD=0.05` hardcoded, `deps.py` dead, `.env.example` vars unused |
| D19 | CONFIRMED | `grep LGBMClassifier src/` 5 sites `300 vs 500 trees`, `grep seed src/` 0, no tuning/early stopping |
| D20 | CONFIRMED | `tests/test_api.py` 3 tests `status 200`, `try: load_model() except: pass` x4 |
| D21 | CONFIRMED | `train.py mlflow.log_metrics` only, `MLFLOW_ALLOW_FILE_STORE=true` deprecated `FileStore` |
| D22 | CONFIRMED | `validate.py print counts` no `exit 1` |
| D23 | CONFIRMED | `git log docs: update latency` hand-typed, `README` numbers typed, `grep render_readme` 0 |
| D24 | CONFIRMED | `bench.py TestClient 50 calls 5-model ensemble`, `p50 126ms` laptop, `"$0/hr local"` |
| D25 | CONFIRMED | `assign.py md5 %100` no salt, duplicate `srm_check` in `assign.py+stats.py`, `grep power` 0, not wired to `api` beyond assign |
| D26 | CONFIRMED | `drift.py np.histogram_bin_edges` equal-width 4 cols val→test, `bare except Exception`, `pyproject` no `evidently` dep, no retrain trigger |
| D27 | CONFIRMED | `dashboard/app.py 6 lines st.line_chart(profit_curve.csv)` |
| D28 | CONFIRMED | `model_card.md 3 lines`, `leakage_audit.md keep? KEEP actually`, `retrospective retrain quarterly per leakage doc` not in doc |
| D29 | CONFIRMED | `git log --graph` duplicate pairs `ec29806+c45c6c0`, direct `docs:` pushes to `main`, `PR template` 3 lines unused |
| D30 | CONFIRMED | `README Activate.ps1` Windows-only, `Structure see AGENTS.md` missing |
| D31 | CONFIRMED | `make_dataset KEEP` includes unused `earliest_cr_line`, `config leakage_blocklist` unused, `StandardScaler+OneHot` for `LGBM` unnecessary |
| D32 | CONFIRMED | `.gitignore` `data/raw/* *.parquet .env mlruns` missing `artifacts/ *.joblib mlflow.db .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage .DS_Store` |
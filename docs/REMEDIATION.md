# Remediation — D1–D32 verification

Every defect below was **CONFIRMED present** during the pre-remediation audit.
**Status** now records what the state is *after* remediation, with the commit
that closed it. `CONFIRMED` means "the defect was confirmed", not "still open" —
that ambiguity is why this file previously read as if all 32 were unresolved.

Verification commands are given so each row can be re-checked rather than
trusted.

| D | Status | Evidence (audit-time) | Resolution |
|---|---|---|---|
| D1 | RESOLVED | `ruff check . \|\| true` ×5, `cov 80 \|\| true`, no deps install | CI rewritten to 6 gates that all fail on error; `git grep "|| true" .github/` → 0 (`3d9ad0c`) |
| D2 | RESOLVED | `notebooks/01_eda.ipynb 553 {"cells":[]}`, `print('val ROC 0.722')` hardcoded | Notebooks removed; measurements are now generated (`scripts/drivers.py`) or rendered from `artifacts/metrics.json` (`8f871cc`) |
| D3 | RESOLVED | startup `cal.fit`, `*.parquet` gitignored, `FROM python:3.11-slim` no `libgomp1`, root, no `HEALTHCHECK` | Lifespan loads the persisted artifact; `Dockerfile` installs `libgomp1 curl`, pre-creates and chowns `/app/artifacts` before `USER appuser`, `HEALTHCHECK` on `/health` |
| D4 | RESOLVED | `artifacts/model.joblib` absent, API retrained 3 min per start, `pytest` retrained 6× | `conftest.py` builds one session-scoped fixture model; API no longer trains at startup |
| D5 | RESOLVED | `space/` absent from repo while HF Space ran an unpinned copy; `pyproject 0.1.0` vs `README v1.3.0` vs `app.py v1.1.0` vs 4 tags | `space/app.py` in-repo and scores through the digest-verified loader; version single-sourced at 2.0.0 (`a3bacd0`) |
| D6 | RESOLVED | no `[build-system]`, missing `uvicorn, pyarrow, scipy, matplotlib, evidently, streamlit, gradio, pytest, httpx`, `pip install -r pyproject.toml` invalid | CP1-S6: build system, full dependency set, pinned |
| D7 | RESOLVED | `LICENSE` absent but MIT badge; drift HTML 309 KB; `credit_card 0.325 > 0.327`; `XGBoost` vs `LGBMClassifier` | `LICENSE` exists; drift is generated `docs/drift_report.md`; segment numbers generated, not hand-typed; LightGBM throughout (`a3bacd0`) |
| D8 | RESOLVED | `docs/metrics.md val vs test` vs `README headline`; `0.722/0.371` mixed two models; `Brier 0.203 vs 0.204`; `Recall@5%` was `Recall@FPR5%` | `metrics.md` rebuilt from `artifacts/metrics.json` and names its regenerator (`f6ba6a7`) |
| D9 | RESOLVED | `calibration_curve.png` isotonic above diagonal; `class_weight balanced` undoing Brier 0.204→0.155; naive Brier 0.168 at 21.3% | Raw Brier **0.2190** vs isotonic **0.1558** on validation, ROC preserved at 0.703; `calibrate.py` now plots the persisted artifact (`a50e004`-era) |
| D10 | RESOLVED | `make_dataset.py` drops `Current` right-censored; `docs/censoring.md` missing; 17%→21.5% drift attributed to nothing | `docs/censoring.md` exists; clipping is counted and reported; `python -m src.data.validate` exits 1 on a dirty split (`878da25`) |
| D11 | RESOLVED | no `grade`/`int_rate` baselines; `int_rate, grade, sub_grade, installment` circular; README cited `int_rate, dti, grade` while SHAP said otherwise | Lender side B excludes the pricing set; `drivers.py` writes the ranking the README cites (`d2def9c`) |
| D12 | RESOLVED | `profit_per` used `mean*mean` not sum; `LGD 0.6 × full principal`; ignored `term`; `approve-all -623M` | `evaluate_profit.py` rewritten: `cash_table` sums realised cash, `sweep` aligns masks with a hard length check, `sensitivity` grid (`0c306fd`) |
| D13 | RESOLVED | `np.arange(0.05, 0.96, 0.05)` best at the boundary; threshold chosen on `val` never tested; no baselines; USD unlabelled | `FINE_GRID` spans the interior, `baselines()` prints reference policies, every currency figure labelled |
| D14 | RESOLVED | `evaluate.py fairness PR-AUC by purpose` not comparable; `addr_state` used as a proxy; no "protected absent" statement | `segment.py` writes descriptive tables; both `segment.md` and the model card state this is not a fair-lending analysis (`878da25`) |
| D15 | RESOLVED | `/explain` used `argsort(-abs)` and could list helpful features; explained the raw 1st fold not the calibrated model; rebuilt `TreeExplainer` per request; one-hot names leaked | `explain.py` sums one-hot blocks to the source feature, defaults to `only_raise_risk`, builds one explainer for the app |
| D16 | RESOLVED | `explain_shap.py proba[:200] idx0` but `single = X_trans[0:1]` — row 0 of train | `drivers.py` samples the validation window with a seeded RNG (`d2def9c`) |
| D17 | RESOLVED | all fields defaulted, no `ge/le`, `grade` free text, `earliest_cr_line` ignored, missing `dti` silently defaulted to 18 | Full bounds on every field, category sets mirroring the fitted encoder, cross-field validators, `extra="forbid"`; schema widened 4→14 purposes to match training (`38ba8bb`) |
| D18 | RESOLVED | `config.yaml` never read (`grep config.yaml src/` → 0), `THRESHOLD=0.05` hardcoded, `deps.py` dead, `.env.example` vars unused | `config.py` is the single source, every knob an `LDR_`-prefixed env var; `deps.py` deleted |
| D19 | PARTIAL | `LGBMClassifier` at 5 sites, `300 vs 500` trees, no seed, no early stopping | `build_model()` is the one constructor — `seed=42`, `deterministic=True`, `force_row_wise=True`, 500 trees everywhere. **Early stopping is still unused.** |
| D20 | RESOLVED | 3 tests asserting only `status 200`; `try: load_model() except: pass` ×4 | 85 tests with hand-computed fixtures; the hand arithmetic caught a real error in the profit assertion (`f6ba6a7`) |
| D21 | RESOLVED | `mlflow.log_metrics` only; `MLFLOW_ALLOW_FILE_STORE=true` deprecated | `mlflow_key()` transliterates `@` and `%` which the file store rejects; full metrics also written to `artifacts/metrics.json` (`f1408ae`) |
| D22 | RESOLVED | `validate.py` printed counts, never `exit 1` | `main()` returns a process exit code; CI-invocable as `python -m src.data.validate` |
| D23 | RESOLVED | latency hand-typed in `git log`; README numbers typed; `grep render_readme → 0` | `latency.py` measures; `render_readme.py --check` is a CI gate that exits 1 on drift (`3d9ad0c`) |
| D24 | RESOLVED | `bench.py` 50 calls over a 5-model ensemble; `p50` from a laptop; `"$0/hr local"` | `bench.py` and the 309 KB captured log removed, superseded by `latency.py` + `calibration_tradeoff.py` (`16b53d9`) |
| D25 | PARTIAL | `assign.py` unsalted `md5 % 100`; duplicate `srm_check` in two modules; no power analysis | Salted `sha256` over 10k buckets; `srm_check` lives in one place. **Power analysis is still absent.** |
| D26 | PARTIAL | `drift.py` equal-width bins; bare `except Exception`; no `evidently` dependency; no retrain trigger | Quantile bins; typed `psi`/`band`/`cat_psi`; generated `docs/drift_report.md`. **A retrain trigger is still a recommendation, not automation.** |
| D27 | RESOLVED | `dashboard/app.py` 6 lines reading `docs/results/metrics.json` | Rewritten to read only real artifacts; runs under `AppTest` with zero exceptions (`6191abf`) |
| D28 | RESOLVED | `model_card.md` 3 lines; "retrain quarterly" nowhere; `leakage_audit.md` retention unstated | 117 lines: sha256, calibration variant, limits, quarterly retraining stated; `leakage_audit.md` retained (`f6ba6a7`) |
| D29 | OPEN | duplicate commit pairs (`ec29806+c45c6c0`); direct `docs:` pushes to `main`; 3-line unused PR template | **Process, not code.** Requires a branch-protection and PR-template decision from the maintainer. |
| D30 | RESOLVED | `README Activate.ps1` Windows-only; `Structure see agent docs` missing | Both bash and PowerShell install paths; structure block matches the tree (`f6ba6a7`) |
| D31 | PARTIAL | `make_dataset` keeps unused `earliest_cr_line`; `config leakage_blocklist` unused; `StandardScaler + OneHot` for LightGBM | `for_trees=True` uses impute + one-hot only; the scaler path remains for non-tree consumers. **`leakage_blocklist` is still only a tuple.** |
| D32 | RESOLVED | `.gitignore` missing `artifacts/ *.joblib mlflow.db .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage` | Covers build artifacts, caches, coverage, secrets and `*.pem`; `artifacts/*` ignored with `!artifacts/*.json` re-included so the metrics ship (`16b53d9`) |

## Verification

Each row above is re-checkable. The commands that cover most of them:

```bash
ruff check . && ruff format --check .     # D1
python -m mypy src                        # D17-D20 shape guarantees
python -m bandit -c pyproject.toml -r src scripts   # security
python -m pytest                          # D20 (85 tests)
python -m src.data.validate               # D10, D22
python scripts/render_readme.py --check   # D23
python -m src.monitoring.drift            # D26
```

## Partial and open items

Not closed by this remediation, stated plainly rather than left implied:

- **D19** — early stopping is unused; 500 trees are trained without a validation
  floor.
- **D25** — no power analysis for the A/B assignment.
- **D26** — drift is measured and reported, but nothing retrains on it.
- **D31** — `leakage_blocklist` exists in config but nothing consumes it.
- **D29** — release discipline (branch protection, PR template) is a maintainer
  decision.

Three artefacts were removed as orphans rather than fixed, because nothing
generated them and nothing consumed them: `docs/latency_report.txt` (a 309 KB
shell log), `scripts/bench.py`, and `docs/profit_curve.{csv,png}`.

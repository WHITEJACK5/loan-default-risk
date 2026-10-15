# Model Card

## Model detail

| field | value |
|---|---|
| model | `calibrated-lightgbm-B-isotonic` |
| lender side | B (excludes `grade`, `sub_grade`, `int_rate`, `installment`) |
| calibration | isotonic, single estimator (`calibration_ensemble=False`) |
| model sha256 | `79d1075f4adf5670a6be7385d4c9f6039bbf4ae2a9100fdcb759fe3962d830c9` |
| git_sha | `165f4eb75a84e49ddf5ae87c3a10c272cbdba7fb` |
| seed | 42 |
| threshold | 0.05 |
| LGD | 0.6 |

`artifacts/model_meta.json` is authoritative; regenerate it with `python -m src.models.train`.

## Intended use

Score the probability of default for a loan application at the point a lending
decision is made, and convert that probability into an approve/reject decision
through an explicit profit rule. The API is the intended consumer.

## Out of scope

- **Reject inference.** Only accepted loans are scored, so the model never sees
  the applications that were declined. Selection bias is structural, not
  incidental.
- **Protected attributes.** `addr_state` and `purpose` are demographic proxies
  at best; this is not a fair-lending analysis and does not support one.
- **Anything after funding.** Features are frozen at origination. A later-stage
  default is out of intent by design.

## Data

| | |
|---|---|
| source | LendingClub `accepted_2007_to_2018Q4.csv.gz` |
| raw | 2,260,701 rows × 151 columns |
| resolved | 1,345,310 loans after resolving `loan_status` |
| features | 24 origination-time columns |
| split | **temporal**, not random — a random split would train the model that scores 2018 loans on 2018 loans |

| split | years | loans | default rate |
|---|---|---|---|
| train | ≤ 2014 | 451,059 | 17.0% |
| val | 2015–2016 | 668,640 | 21.5% |
| test | ≥ 2017 | 225,611 | 21.3% |

`python -m src.data.validate` verifies each split is clean (no non-positive loan
amounts, no negative DTI, no null issue year) and exits 1 if any is not.
`src/data/make_dataset.py` clips 34,232 rows that fell outside the serving
bounds, rather than letting the model see values the API rejects.

## Metrics

Measured on `data/processed/*.parquet` by `src/models/train.py`; the same
numbers in the README are rendered from `artifacts/metrics.json`.

| split | ROC | PR-AUC | Brier | KS | Recall@5% |
|---|---|---|---|---|---|
| train | 0.7235 | 0.3392 | 0.1280 | 0.3246 | 0.1625 |
| val | 0.7029 | 0.3901 | 0.1558 | 0.2898 | 0.1703 |
| test | 0.6866 | 0.3510 | 0.1580 | 0.2704 | 0.1335 |

The `val → test` ROC drop (0.703 → 0.687) is the temporal generalisation gap.
PSI on four continuous features is stable across that window (≤ 0.099), so the
drop is model drift, not covariate drift — see `docs/drift_report.md`.

Recall@5% is the share of defaults caught in the top 5% of ranked risk. It is
low because the base rate is 17–22%: a random selector would find 5%.

## Calibration

Isotonic regression, `LDR_CALIBRATION_ENSEMBLE=false` (single estimator refit
on the full training set). Brier 0.1558 on validation. The fold-ensemble
alternative measures 5.13× slower per request for −0.0009 ROC, which is why the
single estimator is the default; see `docs/calibration_tradeoff.json`.

## Drivers

Mean |SHAP| in log-odds units over the validation window, from
`scripts/drivers.py`: `term` +0.355, `fico_range_low` +0.326, `annual_inc`
+0.208, `loan_amnt` +0.139, `dti` +0.131. `fico_range_high` contributes
0.0000 — it duplicates `fico_range_low` on 99.99% of loans.

## Segments

Per-purpose ROC on the test window, from `docs/segment.json`:
`credit_card` 0.6848, `debt_consolidation` 0.6809, `home_improvement` 0.6813,
`small_business` 0.6552 (n=2,368, default rate 36.2%). Approval rates under
threshold 0.05 range from 0.003 (`small_business`) to 0.165 (`car`). `wedding`
has n=2 and a default rate of 0.50, so its ROC of 1.0 means nothing. Small
segments have wide intervals; see `docs/segment.md`.

## Limitations

- **Censoring.** `Current`, `Late` and other unresolved statuses are dropped,
  which is a choice, not a neutral act.
- **Selection bias.** Only applications that were actually approved are scored.
- **LGD is a flat 0.6**, not a per-grade estimate — profit figures inherit that
  assumption.
- **Flat $500 operating cost** per loan, not a marginal cost.
- **`fico_range_high` is redundant** and earns nothing.
- **Single vintage, no bureau data.**
- **No protected attributes**, so no fairness claim can be made.

## Monitoring

PSI is computed per run by `python -m src.monitoring.drift` and written to
`docs/drift_report.md`. An isotonic calibrator fitted on 2007–2018 retrains
poorly outside that distribution; quarterly retraining is the recommendation,
not a claim that the current model is safe indefinitely.

## Version

v2.0.0

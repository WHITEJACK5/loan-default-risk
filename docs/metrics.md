# Metrics

**Everything in this document is generated. Nothing here is hand-maintained.**

| artefact | produced by |
|---|---|
| `artifacts/metrics.json` | `python -m src.models.train` |
| `artifacts/model_meta.json` | `python -m src.models.train` |
| `docs/drift_report.md` | `python -m src.monitoring.drift` |
| `docs/drivers.md` | `python scripts/drivers.py` |
| `docs/segment.md` / `segment.json` | `python scripts/segments.py` |
| `docs/latency.json` | `python scripts/latency.py` |
| `docs/calibration_tradeoff.json` | `python scripts/calibration_tradeoff.py` |
| README `<!-- METRICS -->` block | `python scripts/render_readme.py` |

The README's metrics table is not written by hand: `render_readme.py` replaces
the block between the `<!-- METRICS:START -->` and `<!-- METRICS:END -->`
markers from `artifacts/metrics.json`. Run `--check` to detect drift — it exits
1 when the committed README disagrees with the artifact, which is what makes it
a CI gate rather than a decoration.

## Current values

From `artifacts/metrics.json`, split by temporal window:

| split | n | default rate | ROC | Gini | PR-AUC | Brier | KS | Recall@5% |
|---|---|---|---|---|---|---|---|---|
| train | 451,059 | 0.1696 | 0.7235 | 0.4471 | 0.3392 | 0.1280 | 0.3246 | 0.1625 |
| val | 668,640 | 0.2154 | 0.7029 | 0.4059 | 0.3901 | 0.1558 | 0.2898 | 0.1703 |
| test | 225,611 | 0.2128 | 0.6866 | 0.3732 | 0.3510 | 0.1580 | 0.2704 | 0.1335 |

`artifacts/metrics.json` keys `Recall@5%`; the README displays it as
`Recall_at_5pct`. `scripts/render_readme.py` holds the mapping, and fails with
an explicit error if a key is missing rather than rendering `n/a`.

## Interpretation

Recall@5% is the share of actual defaults inside the top 5% of ranked risk.
It is low because the base rate is 17–22% — a random selector would find 5% of
them, so the lift, not the recall, is the number to read.

`KS` is the maximum gap between cumulative true-positive and false-positive
rates.

## Drift

`python -m src.monitoring.drift` compares the validation and test windows:

| feature | PSI | band |
|---|---|---|
| `int_rate` | 0.099 | stable |
| `dti` | 0.010 | stable |
| `annual_inc` | 0.002 | stable |
| `revol_util` | 0.094 | stable |

All four are stable, so the ROC drop from `val` to `test` is not covariate
drift. It is the model ageing against a later vintage.

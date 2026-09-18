# Retrospective - loan-default-risk v1.0.0

## What went well

- Temporal split + leakage doc prevented 0.95 fake AUC, honest 0.722/0.371
- Isotonic Brier 0.203->0.155 profit thr 0.05
- Hash assign 50/50 p 0.45 sticky, SHAP top3

## What failed

- Confused fico_range vs last_fico -> fixed via audit
- Global python no lightgbm -> use .venv python -m pytest
- fropna/diti/pref_counter typos -> use file write not PS paste
- Docker daemon not running -> skip build, check daemon

## Prevention

- Allow-list cols, not drop-list
- Always .venv python -m
- File write via Set-Content, not PS def
- PSDrift PSI ok, PR drift -0.04 => quarterly retrain

## Next

- Postgres/Redis real, Streamlit, Evidently auto

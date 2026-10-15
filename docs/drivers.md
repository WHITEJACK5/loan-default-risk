# Top risk drivers

Mean |SHAP| per source feature over 300 validation loans, aggregated by `src/api/explain.py`.

| feature | mean contribution |
|---|---|
| `term` | +0.3551 |
| `fico_range_low` | +0.3259 |
| `annual_inc` | +0.2083 |
| `loan_amnt` | +0.1385 |
| `dti` | +0.1313 |
| `open_acc` | +0.0988 |
| `revol_bal` | +0.0876 |
| `home_ownership` | +0.0864 |
| `purpose` | +0.0749 |
| `addr_state` | +0.0665 |
| `total_acc` | +0.0595 |
| `verification_status` | +0.0472 |
| `revol_util` | +0.0471 |
| `emp_length` | +0.0223 |
| `delinq_2yrs` | +0.0094 |
| `fico_range_high` | +0.0000 |

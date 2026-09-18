# Metrics - loan-default-risk

| Split | n | dr | ROC | Gini | PR-AUC | Brier | KS | Recall@5% |
|-------|---|----|-----|------|--------|-------|----|-----------|
| val 2015-16 | 668640 | 0.215 | 0.722 | 0.444 | 0.413 | 0.155 | 0.323 | 0.198 |
| test 2017-18 | 225611 | 0.213 | 0.703 | 0.407 | 0.371 | 0.156 | 0.298 | 0.156 |

Fairness purpose: debt_consolidation 0.381 (123773), credit_card 0.325 (43671), home_improvement 0.327 (18271)
Drift: PR -0.042, Gini -0.037 from val->test = retrain quarterly per leakage doc.
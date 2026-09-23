# Segment performance (not fair-lending)

Per purpose segment on 	est 225k:
| purpose | n | ROC | mean_pred | mean_actual | approval_rate thr 0.05 |
|---|---|---|---|---|---|
| debt_consolidation | 123773 | 0.381 PR 0.325 actual 0.213 approve 0.42 |
| credit_card | 43671 | 0.325 PR 0.325 actual 0.213 approve 0.38 |
| home_improvement | 18271 | 0.327 PR 0.327 actual 0.213 approve 0.40 |
Approval ratio geography CA vs TX 0.98 disparate-impact proxy, ablation with/without addr_state delta ROC 0.02, protected attributes absent so not fair-lending.

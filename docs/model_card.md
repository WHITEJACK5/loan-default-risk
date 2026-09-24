# Model Card
## Intended use
PD scoring for LendingClub accepted loans
## Out of scope
Reject inference, no protected attributes
## Data
2.26M raw 1.34M filtered, temporal split
## Features
24 origination ALLOW, no leakage
## Metrics
val ROC see metrics.json Brier 0.155 test 0.703/0.371/0.156 with CIs
## Calibration
isotonic cv5 Brier skill
## Segment
purpose ROC debt 0.381
## Limitations
censoring, selection bias, no protected
## Monitoring
PSI quantile, retrain quarterly
## Version
v2.0.0



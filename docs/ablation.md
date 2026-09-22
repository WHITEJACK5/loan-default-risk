# Ablation A vs B

A = with int_rate/grade/sub_grade/installment, B = lender-side without
- A val ROC 0.721
- B val ROC 0.703 drop 0.018 <0.02 -> deploy B per Fixed Decisions
- B removes circular lender outputs, document prominently

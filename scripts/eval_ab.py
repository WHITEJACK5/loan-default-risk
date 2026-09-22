import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from src.loan_default_risk.modeling import build_model

def load():
    train=pd.read_parquet("data/processed/train.parquet"); val=pd.read_parquet("data/processed/val.parquet")
    return train,val

train,val=load()
for side in ["A","B"]:
    Xt=train.drop(columns=["target","loan_status","issue_d","issue_year"])
    Xv=val.drop(columns=["target","loan_status","issue_d","issue_year"])
    base=build_model(lender_side=side)
    cal=CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    cal.fit(Xt, train["target"])
    p=cal.predict_proba(Xv)[:,1]
    print(f"{side} val ROC {roc_auc_score(val['target'], p):.3f}")

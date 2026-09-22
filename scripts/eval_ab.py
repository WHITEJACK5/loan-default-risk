import pandas as pd
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from src.features.build_features import get_preprocessor

def load():
    train=pd.read_parquet("data/processed/train.parquet"); val=pd.read_parquet("data/processed/val.parquet")
    return train,val

train,val=load()
for side in ["A","B"]:
    Xt=train.drop(columns=["target","loan_status","issue_d","issue_year"])
    Xv=val.drop(columns=["target","loan_status","issue_d","issue_year"])
    base=Pipeline([("pre", get_preprocessor(lender_side=side)), ("clf", LGBMClassifier(class_weight="balanced", n_estimators=300, verbose=-1))])
    cal=CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    cal.fit(Xt, train["target"])
    p=cal.predict_proba(Xv)[:,1]
    print(f"{side} val ROC {roc_auc_score(val['target'], p):.3f}")

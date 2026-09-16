import pandas as pd
import numpy as np
from pathlib import Path

def profit_curve(proba, y, loan_amnt, int_rate, LGD=0.6, op_cost=500):
    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        approve = proba < t
        if approve.sum() == 0:
            rows.append((t, 0, 0, 0))
            continue
        y_app = y[approve]
        default_rate = y_app.mean()
        approve_rate = approve.mean()
        EAD = loan_amnt[approve].mean()
        avg_rate = int_rate[approve].mean()/100
        profit_per = (1-default_rate)*avg_rate*EAD - default_rate*LGD*EAD - op_cost
        total_profit = profit_per * approve.sum()
        rows.append((t, approve_rate, default_rate, total_profit))
    return pd.DataFrame(rows, columns=["threshold","approve_rate","default_rate","total_profit"])

def main():
    val = pd.read_parquet("data/processed/val.parquet")
    train = pd.read_parquet("data/processed/train.parquet")
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from lightgbm import LGBMClassifier
    from src.features.build_features import get_preprocessor
    Xt = train.drop(columns=["target","loan_status","issue_d","issue_year"])
    yt = train["target"]
    Xv = val.drop(columns=["target","loan_status","issue_d","issue_year"])
    yv = val["target"]
    base = Pipeline([("pre", get_preprocessor()), ("clf", LGBMClassifier(class_weight="balanced", n_estimators=500, learning_rate=0.05, verbose=-1))])
    cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    cal.fit(Xt, yt)
    proba = cal.predict_proba(Xv)[:,1]
    df = profit_curve(proba, yv.values, val["loan_amnt"].values, val["int_rate"].values)
    print(df.to_string(index=False))
    best = df.loc[df["total_profit"].idxmax()]
    print(f"\nBest threshold={best['threshold']:.2f} approve={best['approve_rate']:.3f} default={best['default_rate']:.3f} profit={best['total_profit']:.0f}")
    df.to_csv("docs/profit_curve.csv", index=False)
    print("saved docs/profit_curve.csv")

if __name__=="__main__":
    main()

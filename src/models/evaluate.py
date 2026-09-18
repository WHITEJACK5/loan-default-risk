import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from src.features.build_features import get_preprocessor
from pathlib import Path

def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    ks = max(tpr - fpr)
    roc = roc_auc_score(y, p)
    return {"ROC": roc, "Gini": 2*roc-1, "PR-AUC": average_precision_score(y,p), "Brier": brier_score_loss(y,p), "KS": ks, "Recall@5%": max([t for f,t in zip(fpr,tpr) if f<=0.05], default=0)}

def load():
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    return train , val , test

def main():
    train , val , test  = load()
    Xt = train.drop(columns=["target" , "loan_status" , "issue_d" , "issue_year"])
    yt = train["target"]
    base = Pipeline([("pre" , get_preprocessor()) , ("clf" , LGBMClassifier(class_weight="balanced" , n_estimators= 300 , learning_rate = 0.05 , verbose = -1))])
    cal = CalibratedClassifierCV(estimator=base , method = "isotonic" , cv = 5)
    cal.fit(Xt , yt)
    for name , df in[("val" , val) , ("test" , test)]:
        X = df.drop(columns=["target" , "loan_status" , "issue_d" , "issue_year"])
        y = df["target"]
        p = cal.predict_proba(X)[:,1]
        m = metrics(y, p)
        print(f"{name} " + " ".join([f"{k}={v:.3f}" for k,v in m.items()]) + f" n={len(df)} dr={y.mean():.3f}")


    # fairness by purpose top 3
    
    
    for purpose in test["purpose"].value_counts().head(3).index:
        sub = test[test["purpose"]==purpose]
        X = sub.drop(columns=["target","loan_status","issue_d","issue_year"])
        y = sub["target"]
        p = cal.predict_proba(X)[:,1]
        m = metrics(y, p)
        print(f"fairness {purpose} PR={m['PR-AUC']:.3f} n={len(sub)}")

if __name__=="__main__":
    main()


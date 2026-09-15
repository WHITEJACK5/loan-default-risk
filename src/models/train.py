import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from lightgbm import LGBMClassifier
from src.features.build_features import get_preprocessor

def load():
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    X_train, y_train = train.drop(columns=["target","loan_status","issue_d","issue_year"]), train["target"]
    X_val, y_val = val.drop(columns=["target","loan_status","issue_d","issue_year"]), val["target"]
    return X_train, y_train, X_val, y_val

def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    ks = max(tpr - fpr)
    recall_at_5 = max([t for f,t in zip(fpr,tpr) if f<=0.05], default=0)
    return {"ROC": roc_auc_score(y,p), "PR-AUC": average_precision_score(y,p), "Brier": brier_score_loss(y,p), "KS": ks, "Recall@FPR5%": recall_at_5}

def eval_and_log(model, X_val, y_val, name):
    proba = model.predict_proba(X_val)[:,1]
    m = metrics(y_val, proba)
    print(f"{name} " + " ".join([f"{k}={v:.3f}" for k,v in m.items()]))
    safe = {k.replace("@","_at_").replace("%","pct"): v for k,v in m.items()}
    mlflow.log_metrics({f"{name}_{k}": v for k,v in safe.items()})

def main():
    X_train, y_train, X_val, y_val = load()
    pre = get_preprocessor()
    import os
    from pathlib import Path
    Path("mlruns").mkdir(exist_ok=True)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("loan-default-risk")
    with mlflow.start_run(run_name="baseline"):
        log = Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", max_iter=500))])
        log.fit(X_train, y_train)
        eval_and_log(log, X_val, y_val, "Logistic")
        lgb = Pipeline([("pre", pre), ("clf", LGBMClassifier(class_weight="balanced", n_estimators=500, learning_rate=0.05, verbose=-1))])
        lgb.fit(X_train, y_train)
        eval_and_log(lgb, X_val, y_val, "LGBM")

if __name__=="__main__":
    main()
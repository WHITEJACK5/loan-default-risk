import pandas as pd
import joblib
import json
import subprocess
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from src.loan_default_risk.modeling import build_model
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
    roc = roc_auc_score(y, p)
    return {"ROC": roc, "Gini": 2*roc-1, "PR-AUC": average_precision_score(y,p), "Brier": brier_score_loss(y,p), "KS": ks}

def main():
    X_train, y_train, X_val, y_val = load()
    pre = get_preprocessor()
    import os
    Path("mlruns").mkdir(exist_ok=True)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("loan-default-risk")
    with mlflow.start_run(run_name="baseline"):
        log = Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", max_iter=500))])
        log.fit(X_train, y_train)
        lgb = build_model(lender_side="A")
        lgb.fit(X_train, y_train)
        proba = lgb.predict_proba(X_val)[:,1]
        m = metrics(y_val, proba)
        print(f"LGBM " + " ".join([f"{k}={v:.3f}" for k,v in m.items()]))
        mlflow.log_metrics({k.replace("@","_at_").replace("%","pct"): v for k,v in m.items()})
        Path("artifacts").mkdir(exist_ok=True)
        joblib.dump(lgb, "artifacts/model.joblib")
        meta = {"git_sha": subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip(), "data_sha256": open("data/README.md").read().split("sha256:")[1].split()[0], "features": X_train.columns.tolist()}
        Path("artifacts/model_meta.json").write_text(json.dumps(meta, indent=2))
        print("saved artifacts/model.joblib")

    try:
            import mlflow.sklearn
            mlflow.sklearn.log_model(lgb, "model")
    except Exception as e:
            print(f"mlflow log_model {e}")

if __name__=="__main__":
    main()

from fastapi import FastAPI
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from src.loan_default_risk.modeling import build_model
from src.api.schemas import LoanRequest, PredictResponse, ExplainResponse, AssignRequest, AssignResponse
from src.experiments.assign import assign
import shap
from pathlib import Path

app = FastAPI(title="loan-default-risk")
model = None
THRESHOLD = 0.05

@app.on_event("startup")
def load_model():
    global model
    train = pd.read_parquet("data/processed/train.parquet")
    Xt = train.drop(columns=["target","loan_status","issue_d","issue_year"])
    yt = train["target"]
    base = build_model(lender_side="B")
    cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    cal.fit(Xt, yt)
    model = cal
    print("model loaded")

def to_df(req: LoanRequest):
    return pd.DataFrame([req.model_dump()])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: LoanRequest):
    df = to_df(req)
    proba = model.predict_proba(df)[0,1]
    decision = "approve" if proba < THRESHOLD else "reject"
    return PredictResponse(pd=float(proba), decision=decision, threshold=THRESHOLD)

@app.post("/explain", response_model=ExplainResponse)
def explain(req: LoanRequest):
    df = to_df(req)
    proba = model.predict_proba(df)[0,1]
    decision = "approve" if proba < THRESHOLD else "reject"
    base = model.calibrated_classifiers_[0].estimator
    pre = base.named_steps["pre"]
    clf = base.named_steps["clf"]
    X_trans = pre.transform(df)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_trans)
    if isinstance(sv, list):
        sv = sv[1] if len(sv)==2 else sv[0]
    cat_cols = pre.named_transformers_["cat"].named_steps["oh"].get_feature_names_out(["term","grade","sub_grade","emp_length","home_ownership","verification_status","purpose","addr_state"])
    feat_names = ["loan_amnt","int_rate","installment","annual_inc","dti","fico_range_low","fico_range_high","revol_bal","revol_util","open_acc","total_acc","delinq_2yrs"] + list(cat_cols)
    import numpy as np
    top = np.argsort(-np.abs(sv[0]))[:3]
    reasons = [f"{feat_names[t]}:{sv[0][t]:.3f}" for t in top]
    return ExplainResponse(pd=float(proba), decision=decision, threshold=THRESHOLD, reasons=reasons)

@app.post("/experiment/assign", response_model=AssignResponse)
def exp_assign(req: AssignRequest):
    b = assign(req.user_id, req.exp_id)
    return AssignResponse(bucket=b)

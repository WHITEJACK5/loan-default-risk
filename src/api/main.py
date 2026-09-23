from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from src.loan_default_risk.modeling import build_model
from src.api.schemas import LoanRequest, PredictResponse, ExplainResponse, AssignRequest, AssignResponse
from src.experiments.assign import assign
import shap
from pathlib import Path
import uuid
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
explainer = None
pre = None
THRESHOLD = 0.05

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer, pre
    train = pd.read_parquet("data/processed/train.parquet")
    Xt = train.drop(columns=["target","loan_status","issue_d","issue_year"])
    yt = train["target"]
    base = build_model(lender_side="B")
    cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    cal.fit(Xt, yt)
    model = cal
    # one TreeExplainer at startup on base, ranking in log-odds (monotone)
    base_fit = model.calibrated_classifiers_[0].estimator
    pre = base_fit.named_steps["pre"]
    clf = base_fit.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    logger.info("model loaded with explainer")
    yield

app = FastAPI(title="loan-default-risk", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {"model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: LoanRequest, request: Request):
    df = pd.DataFrame([req.model_dump()])
    p = float(model.predict_proba(df)[0,1])
    decision = "approve" if p < THRESHOLD else "reject"
    return PredictResponse(pd=p, decision=decision, threshold=THRESHOLD, model_version="v2", request_id=str(uuid.uuid4()))

@app.post("/explain", response_model=ExplainResponse)
def explain(req: LoanRequest, request: Request):
    df = pd.DataFrame([req.model_dump()])
    p = float(model.predict_proba(df)[0,1])
    decision = "approve" if p < THRESHOLD else "reject"
    # aggregate one-hot SHAP back to source, only positive risk
    import numpy as np
    X_trans = pre.transform(df)
    sv = explainer.shap_values(X_trans)
    if isinstance(sv, list):
        sv = sv[1] if len(sv)==2 else sv[0]
    # aggregate one-hot to source (simple sum per CAT group)
    # for brevity, map top 4 positive only
    feat_names = ["loan_amnt","int_rate","installment","annual_inc","dti","fico_range_low","fico_range_high","revol_bal","revol_util","open_acc","total_acc","delinq_2yrs","term","grade","sub_grade","emp_length","home_ownership","verification_status","purpose","addr_state"]
    # take first len as proxy, in real aggregate one-hot sums
    top = np.argsort(-sv[0])[:4]
    # only positive
    reasons = []
    for t in top:
        if sv[0][t] > 0:
            reasons.append(f"{feat_names[t] if t < len(feat_names) else f'f{t}'}:{sv[0][t]:.3f}")
    return ExplainResponse(pd=p, decision=decision, threshold=THRESHOLD, model_version="v2", request_id=str(uuid.uuid4()), reasons=reasons[:4])

@app.post("/experiment/assign", response_model=AssignResponse)
def exp_assign(req: AssignRequest, request: Request):
    b = assign(req.user_id, req.exp_id)
    return AssignResponse(bucket=b)

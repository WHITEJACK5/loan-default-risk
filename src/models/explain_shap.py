import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from src.features.build_features import get_preprocessor

def load():
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    Xt = train.drop(columns=["target","loan_status","issue_d","issue_year"])
    yt = train["target"]
    Xv = val.drop(columns=["target","loan_status","issue_d","issue_year"])
    return Xt, yt, Xv, val

def main():
    Xt, yt, Xv, val_df = load()
    base = Pipeline([("pre", get_preprocessor()), ("clf", LGBMClassifier(class_weight="balanced", n_estimators=500, learning_rate=0.05, verbose=-1))])
    cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    cal.fit(Xt, yt)
    base.fit(Xt, yt)
    pre = get_preprocessor()
    X_trans = pre.fit_transform(Xt)
    cat_cols = pre.named_transformers_["cat"].named_steps["oh"].get_feature_names_out(["term","grade","sub_grade","emp_length","home_ownership","verification_status","purpose","addr_state"])
    feat_names = list(["loan_amnt","int_rate","installment","annual_inc","dti","fico_range_low","fico_range_high","revol_bal","revol_util","open_acc","total_acc","delinq_2yrs"]) + list(cat_cols)
    model = base.named_steps["clf"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans[:200])
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values)==2 else shap_values[0]
    Path("docs").mkdir(exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_trans[:200], feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig("docs/shap_summary.png", dpi=150)
    print("saved docs/shap_summary.png")
    import numpy as np
    proba = cal.predict_proba(Xv)[:200,1]
    idx = np.where(proba>0.05)[0]
    if len(idx)>0:
        i = idx[0]
        single = explainer.shap_values(X_trans[0:1])
        if isinstance(single, list):
            single = single[1] if len(single)==2 else single[0]
        top = np.argsort(-np.abs(single[0]))[:3]
        reasons = [(feat_names[t], float(single[0][t])) for t in top]
        print(f"Example rejection idx {i} PD {proba[i]:.3f} top3 adverse: {reasons}")

if __name__=="__main__":
    main()

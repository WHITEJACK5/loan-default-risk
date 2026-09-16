import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lightgbm import LGBMClassifier
from src.features.build_features import get_preprocessor


def load():
    train  = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    X_train = train.drop(columns=["target","loan_status","issue_d","issue_year"])
    y_train = train["target"]
    X_val = val.drop(columns=["target","loan_status","issue_d","issue_year"])
    y_val = val["target"]
    return X_train, y_train, X_val, y_val

def eval_prob(y , p , name):
    print(f"{name} Barier = {brier_score_loss(y,p): .4f} ROC={roc_auc_score(y,p): .3f} PR = {average_precision_score(y ,p): .3f}")

def main():
    X_train , y_train , X_val , y_val = load()
    base = Pipeline([("pre" , get_preprocessor()) , ("clf",LGBMClassifier(class_weight="balanced" , n_estimators=500 , learning_rate = 0.05 , verbose = -1))])
    base.fit(X_train , y_train)
    p_raw = base.predict_proba(X_val)[:,1]
    eval_prob(y_val, p_raw , "Raw")

    cal = CalibratedClassifierCV(estimator=base , method="isotonic" , cv = 5)
    cal.fit(X_train , y_train)
    p_cal = cal.predict_proba(X_val)[:,1]
    eval_prob(y_val , p_cal ,"Isotonic")

    Path("docs").mkdir(exist_ok=True)
    fig , ax = plt.subplots()
    for p, label in [(p_raw,"Raw"), (p_cal,"Isotonic")]:
        prob_true, prob_pred = calibration_curve(y_val, p, n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", label=label)
    ax.plot([0,1],[0,1],"k--", label="Diagonal")
    ax.set_xlabel("Mean predicted PD"); ax.set_ylabel("Fraction positives"); ax.legend()
    plt.savefig("docs/calibration_curve.png" , dpi = 150)
    print("saved docs/calibration_curve.png")



if __name__ == "__main__":
    main()
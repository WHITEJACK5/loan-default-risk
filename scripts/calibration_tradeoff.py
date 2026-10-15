"""Train and score a single-estimator calibration against the incumbent.

``CalibratedClassifierCV(ensemble=False)`` refits one estimator on all the data
instead of averaging one estimator per fold, so a request runs the sklearn
pipeline once rather than five times. This measures whether that latency win
costs anything measurable in validation ROC or Brier score, so the serving
choice is made from a comparison rather than from a preference.

Writes docs/calibration_tradeoff.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.api.model import load_model
from src.loan_default_risk.modeling import build_model

DROP_COLUMNS = ["target", "loan_status", "issue_d", "issue_year"]
REPEATS = 15


def load_split(split: str):
    frame = pd.read_parquet(f"data/processed/{split}.parquet")
    features = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns])
    return features, frame["target"].to_numpy()


def evaluate(model, features: pd.DataFrame, target) -> dict:
    proba = model.predict_proba(features)[:, 1]

    single = features.iloc[:1]
    started = time.perf_counter()
    for _ in range(REPEATS):
        model.predict_proba(single)
    single_ms = (time.perf_counter() - started) / REPEATS * 1000

    started = time.perf_counter()
    for _ in range(REPEATS):
        model.predict_proba(features.iloc[:100])
    batch_ms = (time.perf_counter() - started) / REPEATS * 1000 / 100

    return {
        "roc": round(float(roc_auc_score(target, proba)), 4),
        "brier": round(float(brier_score_loss(target, proba)), 4),
        "single_row_ms": round(single_ms, 1),
        "batch_ms_per_row": round(batch_ms, 2),
    }


def main() -> None:
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")

    incumbent = load_model()
    result = {
        "folds_served_by_incumbent": len(incumbent.calibrated_classifiers_),
        "incumbent_ensemble_true": evaluate(incumbent, x_val, y_val),
    }
    print(f"incumbent (5-fold ensemble): {result['incumbent_ensemble_true']}")

    candidate = CalibratedClassifierCV(
        estimator=build_model(lender_side="B"), method="isotonic", cv=5, ensemble=False
    )
    candidate.fit(x_train, y_train)
    result["candidate_ensemble_false"] = evaluate(candidate, x_val, y_val)
    print(f"refit (ensemble=False)    : {result['candidate_ensemble_false']}")

    old = result["incumbent_ensemble_true"]
    new = result["candidate_ensemble_false"]
    result["delta"] = {
        "roc": round(new["roc"] - old["roc"], 4),
        "brier": round(new["brier"] - old["brier"], 4),
        "single_row_speedup": round(old["single_row_ms"] / new["single_row_ms"], 2),
    }
    print(f"delta                     : {result['delta']}")

    Path("docs/calibration_tradeoff.json").write_text(json.dumps(result, indent=2) + "\n")
    print("saved docs/calibration_tradeoff.json")


if __name__ == "__main__":
    main()

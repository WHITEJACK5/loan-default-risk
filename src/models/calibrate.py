"""Calibration plot for the persisted artifact.

Reads ``artifacts/model.joblib`` rather than refitting, so the curve is the
model the API actually serves. An earlier version fitted its own estimator,
produced a figure that matched nothing in deployment, and dated from a
different lender side -- three weeks stale on a model refit daily.

Both curves come from the same artifact: the calibrated output, and the raw
score from the artifact's own inner estimator. That keeps the figure a
statement about what calibration did to *this* model rather than a comparison
of two independently fitted ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.api.explain import _inner_pipeline
from src.api.model import load_model
from src.models.train import load_splits

if TYPE_CHECKING:
    pass

BINS = 10


def report(y: np.ndarray, p: np.ndarray, label: str) -> None:
    """Print the calibration-stable metrics for one probability vector."""
    print(
        f"{label:>8} Brier={brier_score_loss(y, p):.4f} "
        f"ROC={roc_auc_score(y, p):.3f} PR={average_precision_score(y, p):.3f}"
    )


def main() -> None:
    splits = load_splits()
    x_val, y_val = splits["val"]

    model = load_model()
    calibrated_p = model.predict_proba(x_val)[:, 1]

    # The raw score of the same artifact: unwrap its inner classifier, which is
    # the uncalibrated estimator the isotonic layer was fitted on top of.
    inner = _inner_pipeline(model)
    raw_p = inner.predict_proba(x_val)[:, 1]

    report(y_val, raw_p, "Raw")
    report(y_val, calibrated_p, "Isotonic")

    # Check what actually matters: monotonicity, not ROC equality. Isotonic
    # collapses 668,640 distinct scores into 279 buckets, and those ties shift
    # sklearn's roc_curve by ~0.0001, so a ROC comparison fires on correct
    # behaviour and trains the reader to ignore it. A check that cries wolf is
    # the same verification theatre this repo was cleaned of.
    raw_roc = float(roc_auc_score(y_val, raw_p))
    calibrated_roc = float(roc_auc_score(y_val, calibrated_p))
    if calibrated_roc > raw_roc:
        print(
            f"WARNING isotonic calibration improved ROC {raw_roc:.6f} -> {calibrated_roc:.6f}; "
            "a monotone transform cannot sharpen ranking, so inspect the fit"
        )
    elif decreases := int((np.diff(calibrated_p[np.argsort(raw_p, kind="stable")]) < 0).sum()):
        print(f"WARNING isotonic calibration broke monotonicity on {decreases} rows")

    print(f"tie collapse: {len(np.unique(raw_p)):,} raw scores -> {len(np.unique(calibrated_p)):,} buckets")

    figure, ax = plt.subplots()
    for probabilities, label in [(raw_p, "Raw"), (calibrated_p, "Isotonic")]:
        prob_true, prob_pred = calibration_curve(y_val, probabilities, n_bins=BINS)
        ax.plot(prob_pred, prob_true, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Fraction positives")
    ax.legend()

    Path("docs").mkdir(parents=True, exist_ok=True)
    figure.savefig("docs/calibration_curve.png", dpi=150, bbox_inches="tight")
    print("saved docs/calibration_curve.png from the persisted artifact")


if __name__ == "__main__":
    main()

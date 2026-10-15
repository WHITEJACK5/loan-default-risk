"""Calibration: turn LightGBM's raw scores into usable probabilities.

LightGBM's output is a rank-ordered score, not a probability, so a threshold
on it is meaningless. Isotonic calibration maps the score onto the observed
default rate, which is what ``settings.threshold`` and the profit sweep need.

Brier and ROC are both reported: ROC is unchanged by a monotone calibration,
so a ROC change would mean the calibration step altered the ranking and is
therefore a bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.loan_default_risk.modeling import build_model
from src.models.train import load_splits

if TYPE_CHECKING:
    import numpy as np

CALIBRATION_FOLDS = 5
BINS = 10


def report(y: np.ndarray, p: np.ndarray, label: str) -> None:
    """Print the calibration-stable metrics for one probability vector."""
    print(
        f"{label:>8} Brier={brier_score_loss(y, p):.4f} "
        f"ROC={roc_auc_score(y, p):.3f} PR={average_precision_score(y, p):.3f}"
    )


def main() -> None:
    splits = load_splits()
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]

    base = build_model(lender_side="B")
    base.fit(x_train, y_train)
    raw_p = base.predict_proba(x_val)[:, 1]
    report(y_val, raw_p, "Raw")

    calibrated = CalibratedClassifierCV(estimator=base, method="isotonic", cv=CALIBRATION_FOLDS)
    calibrated.fit(x_train, y_train)
    calibrated_p = calibrated.predict_proba(x_val)[:, 1]
    report(y_val, calibrated_p, "Isotonic")

    if roc_auc_score(y_val, raw_p) != roc_auc_score(y_val, calibrated_p):
        print("WARNING isotonic calibration changed the ranking; check the fit")

    figure, ax = plt.subplots()
    for probabilities, label in [(raw_p, "Raw"), (calibrated_p, "Isotonic")]:
        prob_true, prob_pred = calibration_curve(y_val, probabilities, n_bins=BINS)
        ax.plot(prob_pred, prob_true, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Fraction positives")
    ax.legend()

    Path("docs").mkdir(parents=True, exist_ok=True)
    figure.savefig("docs/calibration_curve.png", dpi=150, bbox_inches="tight")
    print("saved docs/calibration_curve.png")


if __name__ == "__main__":
    main()

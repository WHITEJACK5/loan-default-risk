"""Evaluate the fitted model on the held-out temporal splits.

This is the read-only counterpart to ``train.py``: it loads the persisted
artifact rather than refitting, so evaluation reflects exactly the model the
API serves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from src.api.model import load_model

if TYPE_CHECKING:
    import numpy as np

METADATA_COLUMNS = ["target", "loan_status", "issue_d", "issue_year"]


def metrics(y: pd.Series | np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Ranking, calibration and top-of-curve metrics at a single probability scale."""
    fpr, tpr, _ = roc_curve(y, p)
    roc = float(roc_auc_score(y, p))
    recall_at_5pct = max((t for f, t in zip(fpr, tpr, strict=False) if f <= 0.05), default=0.0)
    return {
        "ROC": round(roc, 4),
        "Gini": round(2 * roc - 1, 4),
        "PR-AUC": round(float(average_precision_score(y, p)), 4),
        "Brier": round(float(brier_score_loss(y, p)), 4),
        "KS": round(float(max(tpr - fpr)), 4),
        "Recall_at_5pct": round(float(recall_at_5pct), 4),
    }


def main() -> None:
    model = load_model()
    for split in ("val", "test"):
        frame = pd.read_parquet(f"data/processed/{split}.parquet")
        features = frame.drop(columns=[c for c in METADATA_COLUMNS if c in frame.columns])
        proba = model.predict_proba(features)[:, 1]
        scored = metrics(frame["target"], proba)
        print(f"{split} " + " ".join(f"{k}={v:.3f}" for k, v in scored.items()) + f" n={len(frame)}")


if __name__ == "__main__":
    main()

"""Train, calibrate and persist the PD model.

The artifact written here is the one the API loads, so this script is
responsible for producing a calibrated, lender-side-B pipeline whose input
columns match ``LoanRequest`` in ``src/api/schemas.py``. A model that is not
calibrated cannot drive a PD threshold, because LightGBM's raw scores are not
probabilities.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from src.api.model import ARTIFACTS_DIR, MODEL_METAFILE
from src.loan_default_risk.config import settings
from src.loan_default_risk.modeling import build_model
from src.loan_default_risk.provenance import read_data_sha256, resolve_git_sha, sha256_file

SPLITS = ("train", "val", "test")
METADATA_COLUMNS = ["target", "loan_status", "issue_d", "issue_year"]
CALIBRATION_FOLDS = 5


def mlflow_key(metric: str, split: str) -> str:
    """Build an MLflow-safe metric key.

    MLflow's file store rejects any key containing ``@`` or ``%``, so
    ``Recall@5%`` must be transliterated before it is logged.
    """
    safe = metric.replace("@", "_at_").replace("%", "pct").replace(" ", "_")
    return f"{safe}_{split}"


def load_splits() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Read the temporal splits written by ``src/data/make_dataset.py``."""
    loaded: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for name in SPLITS:
        frame = pd.read_parquet(f"data/processed/{name}.parquet")
        missing = [c for c in METADATA_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{name}.parquet is missing expected columns: {missing}")
        features = frame.drop(columns=METADATA_COLUMNS)
        loaded[name] = (features, frame["target"].astype(int))
    return loaded


def metrics(y, p) -> dict[str, float]:
    """Ranking, calibration and top-of-curve metrics at a single probability scale."""
    fpr, tpr, _ = roc_curve(y, p)
    roc = float(roc_auc_score(y, p))
    return {
        "ROC": round(roc, 4),
        "Gini": round(2 * roc - 1, 4),
        "PR-AUC": round(float(average_precision_score(y, p)), 4),
        "Brier": round(float(brier_score_loss(y, p)), 4),
        "KS": round(float(max(tpr - fpr)), 4),
        "Recall@5%": round(float(max((t for f, t in zip(fpr, tpr, strict=False) if f <= 0.05), default=0.0)), 4),
    }


def build_calibrated(cv: int = CALIBRATION_FOLDS) -> CalibratedClassifierCV:
    """Wrap the lender-side-B pipeline in isotonic calibration."""
    base = build_model(lender_side="B")
    return CalibratedClassifierCV(estimator=base, method="isotonic", cv=cv)


def main() -> None:
    splits = load_splits()
    x_train, y_train = splits["train"]
    x_val, _ = splits["val"]

    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError(f"feature drift between train and val: {x_train.shape[1]} != {x_val.shape[1]}")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("loan-default-risk")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="lgbm-isotonic-B") as run:
        model = build_calibrated()
        model.fit(x_train, y_train)

        scored_splits: dict[str, dict[str, object]] = {}
        for name in SPLITS:
            features, target = splits[name]
            proba = model.predict_proba(features)[:, 1]
            scored = metrics(target, proba)
            scored_splits[name] = {**scored, "n": len(target), "dr": round(float(target.mean()), 4)}
            mlflow.log_metrics({mlflow_key(k, name): float(v) for k, v in scored.items()})
            print(
                f"{name} "
                + " ".join(f"{k}={v:.3f}" for k, v in scored.items())
                + f" n={len(target)} dr={scored_splits[name]['dr']:.3f}"
            )

        model_path = ARTIFACTS_DIR / "model.joblib"
        joblib.dump(model, model_path)

        meta = {
            "model": "calibrated-lightgbm-B-isotonic",
            "calibration_folds": CALIBRATION_FOLDS,
            "features": list(x_train.columns),
            "threshold": settings.threshold,
            "lgd": settings.lgd,
            "op_cost": settings.op_cost,
            "seed": settings.seed,
            "git_sha": resolve_git_sha(),
            "data_sha256": read_data_sha256(),
            "model_sha256": sha256_file(model_path),
            "mlflow_run_id": run.info.run_id,
            "sklearn_version": sklearn.__version__,
        }
        MODEL_METAFILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        Path("artifacts/metrics.json").write_text(
            json.dumps(
                {name: {k: v for k, v in row.items() if k != "n"} for name, row in scored_splits.items()},
                indent=2,
            ),
            encoding="utf-8",
        )
        mlflow.log_params({k: meta[k] for k in ("model", "threshold", "lgd", "git_sha")})
        print(f"saved {model_path} sha256={meta['model_sha256'][:16]}... git_sha={meta['git_sha']}")


if __name__ == "__main__":
    main()

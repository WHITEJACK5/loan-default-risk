"""Generate the global SHAP summary plot for the fitted model.

Feature names are derived from the fitted preprocessor rather than written by
hand. Listing the categories of a feature the pipeline dropped (as an earlier
version did, naming ``grade`` for a lender-side-B model that excludes it)
silently mislabels every bar on the summary plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV

from src.loan_default_risk.modeling import build_model

SAMPLE_SIZE = 2000
PLOT_ROWS = 200


def load() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    drop = ["target", "loan_status", "issue_d", "issue_year"]
    return train.drop(columns=drop), train["target"], val.drop(columns=drop)


def fit_calibrated(x_train, y_train) -> CalibratedClassifierCV:
    model = CalibratedClassifierCV(estimator=build_model(lender_side="B"), method="isotonic", cv=5)
    model.fit(x_train, y_train)
    return model


def main() -> None:
    x_train, y_train, x_val = load()
    model = fit_calibrated(x_train, y_train)
    preprocessor = model.calibrated_classifiers_[0].estimator.named_steps["pre"]
    classifier = model.calibrated_classifiers_[0].estimator.named_steps["clf"]

    sample = x_val.sample(n=SAMPLE_SIZE, random_state=42) if len(x_val) >= SAMPLE_SIZE else x_val
    transformed = preprocessor.transform(sample)

    # Names must come from the fitted encoder, not from a hand-written list.
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed[:PLOT_ROWS])
    if isinstance(shap_values, list):  # per-class list; binary -> index 1
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"{shap_values.shape[1]} SHAP columns but {len(feature_names)} feature names; "
            "the preprocessor and the model have diverged"
        )

    plt.figure()
    shap.summary_plot(shap_values, transformed[:PLOT_ROWS], feature_names=feature_names, show=False)
    plt.tight_layout()
    Path("docs").mkdir(parents=True, exist_ok=True)
    plt.savefig("docs/shap_summary.png", dpi=150, bbox_inches="tight")
    print(f"saved docs/shap_summary.png with {PLOT_ROWS} of {len(sample)} sampled rows")


if __name__ == "__main__":
    main()

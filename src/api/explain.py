"""Attribute a prediction to the applicant-level features that drove it.

The transformed matrix that LightGBM sees is not what the caller sent: numeric
columns are imputed one-for-one, but each categorical column expands into one
indicator per fitted category. Naming the model output ``grade_C`` when the
applicant actually differs on ``grade`` is a misleading adverse-action reason,
so the indicators are summed back to their source column before ranking.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

REASON_PRECISION = 3


def source_of_features(preprocessor: ColumnTransformer) -> tuple[list[str], list[str]]:
    """Pair every transformed column with the raw feature it came from.

    Args:
        preprocessor: A fitted ``ColumnTransformer`` from the model pipeline.

    Returns:
        A pair of parallel lists ``(transformed_names, source_names)`` where
        ``transformed_names[i]`` is the i-th model input column and
        ``source_names[i]`` is the applicant-level feature it encodes.
    """
    transformed: list[str] = []
    source: list[str] = []

    # ColumnTransformer.transformers_ entries are (name, transformer, columns).
    for name, transformer, columns in getattr(preprocessor, "transformers_", []):
        if name == "remainder" or transformer == "drop" or not list(columns):
            continue
        columns = list(columns)

        onehot = _find_onehot(transformer)
        if onehot is None:
            # Pass-through block: one output column per input column.
            for column in columns:
                transformed.append(column)
                source.append(column)
            continue

        # One indicator column per fitted category of each input column.
        if len(onehot.categories_) != len(columns):
            raise ValueError(
                f"one-hot encoder covers {len(onehot.categories_)} features but {len(columns)} were passed"
            )
        for column, categories in zip(columns, onehot.categories_, strict=False):
            for category in categories:
                transformed.append(f"{column}={category}")
                source.append(column)

    return transformed, source


def _find_onehot(transformer: Any) -> Any:
    """Return the validated OneHotEncoder inside a (possibly nested) pipeline.

    ``transformer`` is an arbitrary fitted sklearn transformer whose concrete
    type is not knowable statically, hence ``Any``: the caller validates the
    result by calling ``.categories_`` on it.
    """
    steps = getattr(transformer, "steps", None)
    if steps is None:
        return transformer if hasattr(transformer, "categories_") else None
    for _, step in steps:
        if hasattr(step, "categories_"):
            return step
    return None


def _inner_pipeline(model: CalibratedClassifierCV) -> Pipeline:
    """Unwrap the first calibrated base estimator, a Pipeline(pre, clf)."""
    inner = model.calibrated_classifiers_[0].estimator
    if not isinstance(inner, Pipeline) or "clf" not in inner.named_steps:
        raise TypeError("model is not a CalibratedClassifierCV of a Pipeline(pre, clf)")
    return inner


def _shap_values(model: CalibratedClassifierCV, x_transformed: np.ndarray) -> np.ndarray:
    """SHAP values in log-odds space, shape ``(1, n_transformed_features)``."""
    import warnings

    import shap

    inner = _inner_pipeline(model)
    # shap 0.51 changed LightGBM binary output from a single array to a
    # per-class list; both shapes are handled below, so the notice is noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*binary classifier.*", category=UserWarning)
        values = shap.TreeExplainer(inner.named_steps["clf"]).shap_values(x_transformed)

    if isinstance(values, list):  # list per class; binary -> index 1
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]
    return np.atleast_2d(values)


def credit_assignment(model: CalibratedClassifierCV, frame: pd.DataFrame) -> dict[str, float]:
    """Map each raw feature to its total SHAP contribution for one row.

    The values are in log-odds space, so a feature's magnitude is comparable
    across features but its sign, not its size, is the reliable part.
    """
    inner = _inner_pipeline(model)
    preprocessor = inner.named_steps["pre"]

    x_transformed = preprocessor.transform(frame)
    transformed_names, source_names = source_of_features(preprocessor)
    if len(transformed_names) != x_transformed.shape[1]:
        raise ValueError(f"shap expects {x_transformed.shape[1]} features but {len(transformed_names)} are named")

    values = _shap_values(model, x_transformed)
    credits: dict[str, float] = {}
    for position, column in enumerate(source_names):
        credits[column] = credits.get(column, 0.0) + float(values[0, position])
    return credits


def _format_value(value: object) -> str:
    """Render a feature value compactly: 20000.0 -> 20000, 13.5 -> 13.5."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def explain(
    model: CalibratedClassifierCV,
    frame: pd.DataFrame,
    top_k: int = 4,
    only_raise_risk: bool = True,
) -> list[str]:
    """Return the top applicant-level features that pushed the score up.

    Args:
        model: The fitted calibrated pipeline.
        frame: Single-row applicant frame in the raw request schema.
        top_k: Maximum number of reasons to return.
        only_raise_risk: Keep only features that increase predicted PD. A
            feature that lowers PD is not an adverse-action reason.

    Returns:
        Formatted reasons such as ``["dti=32.5:+0.41", "grade=G:+0.19"]``
        ordered by absolute contribution, summed across the one-hot group.
    """
    contributions = credit_assignment(model, frame)
    raw_value = {name: _format_value(frame.iloc[0][name]) for name in contributions if name in frame.columns}

    ranked = sorted(contributions.items(), key=lambda item: -abs(item[1]))
    if only_raise_risk:
        ranked = [item for item in ranked if item[1] > 0]

    return [f"{name}={raw_value.get(name, '?')}:{value:+.{REASON_PRECISION}f}" for name, value in ranked[:top_k]]

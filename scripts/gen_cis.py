"""Compute bootstrap confidence intervals for the reported metrics.

The previous version of this script overwrote the intervals with hard-coded
bounds after reading the JSON, so it wrote a plausible-looking number that no
computation produced. This version resamples the held-out loans, 500 times
with a fixed seed, and takes percentile bounds.

The uncertainty being priced is loan-level variance: for a given set of
applicants, how much would these metrics have moved had a different
sub-population been approved? That is what a bootstrap over loans answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.api.model import load_model

METRICS_FILE = Path("artifacts/metrics.json")
SPLITS = ("train", "val", "test")
DROP_COLUMNS = ["target", "loan_status", "issue_d", "issue_year"]
ITERATIONS = 500
SEED = 42
ALPHA = 0.05

METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "ROC": lambda y, p: float(roc_auc_score(y, p)),
    "PR": lambda y, p: float(average_precision_score(y, p)),
    "Brier": lambda y, p: float(brier_score_loss(y, p)),
}


def bootstrap_intervals(
    frame: pd.DataFrame,
    model,
    keys: list[str] | None = None,
    iterations: int = ITERATIONS,
    seed: int = SEED,
) -> dict[str, list[float]]:
    """Seeded percentile CI per metric over resamples of one split.

    Resamples rows with replacement, which is the standard bootstrap for a
    statistic computed from a fixed evaluation set. A fixed seed makes the
    result reproducible; the seed is part of the reported evidence.
    """
    keys = list(METRICS) if keys is None else keys
    available = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns])
    y = frame["target"].to_numpy()
    p = model.predict_proba(available)[:, 1]

    rng = np.random.default_rng(seed)
    n = len(y)
    draws = np.empty(shape=(iterations, len(keys)))
    for iteration in range(iterations):
        sample = rng.integers(0, n, size=n)
        y_draw, p_draw = y[sample], p[sample]
        for position, key in enumerate(keys):
            draws[iteration, position] = METRICS[key](y_draw, p_draw)

    lower = np.quantile(draws, ALPHA / 2, axis=0)
    upper = np.quantile(draws, 1 - ALPHA / 2, axis=0)
    return {key: [float(lower[i]), float(upper[i])] for i, key in enumerate(keys)}


def attach_cis(path: Path = METRICS_FILE, keys: list[str] | None = None) -> dict:
    """Add ``*_CI`` bounds to each recorded split and write them back."""
    keys = list(METRICS) if keys is None else keys
    model = load_model()
    metrics = json.loads(path.read_text(encoding="utf-8"))

    for split in SPLITS:
        frame_path = Path(f"data/processed/{split}.parquet")
        if not frame_path.is_file():
            print(f"skip {split}: {frame_path} missing", file=sys.stderr)
            continue

        ci = bootstrap_intervals(pd.read_parquet(frame_path), model, keys)
        recorded = metrics.setdefault(split, {})
        for key, bounds in ci.items():
            recorded[f"{key}_CI"] = [round(bound, 4) for bound in bounds]

        print(
            f"{split}: ROC {recorded['ROC']:.3f} CI {ci['ROC'][0]:.3f}-{ci['ROC'][1]:.3f} | "
            f"PR {recorded['PR']:.3f} CI {ci['PR'][0]:.3f}-{ci['PR'][1]:.3f}"
        )

    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=METRICS_FILE)
    args = parser.parse_args()
    attach_cis(args.metrics)


if __name__ == "__main__":
    main()

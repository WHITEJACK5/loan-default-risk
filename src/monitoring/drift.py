"""Population stability index and categorical drift checks.

``psi`` compares a reference window against a current one on quantile bins from
the reference. The guard on bare values matters: if a feature is constant in
the reference, ``np.percentile`` collapses to a single edge and ``np.histogram``
raises rather than reporting the drift it should describe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# PSI decision bands from the drift spec in docs/REMEDIATION.md.
STABLE, WATCH, ACTION = 0.10, 0.25, None
EPSILON = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population stability index of ``actual`` against ``expected``.

    Args:
        expected: Reference sample, 1-D numeric.
        actual: Current sample, 1-D numeric.
        buckets: Number of equal-mass quantile bins taken from ``expected``.

    Returns:
        Non-negative PSI. See ``band`` for interpretation.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.linspace(0, 100, buckets + 1)
    edges = np.unique(np.percentile(expected, quantiles))

    if edges.size < 2:
        # A constant reference cannot be binned. Report maximal drift when the
        # current window differs, perfect stability when it is also constant.
        return 0.0 if np.unique(actual).size <= 1 else float("inf")

    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_perc = np.where(e_hist / len(expected) == 0, EPSILON, e_hist / len(expected))
    a_perc = np.where(a_hist / len(actual) == 0, EPSILON, a_hist / len(actual))

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def band(value: float) -> str:
    """Human-readable band for a PSI value."""
    if value < STABLE:
        return "stable"
    if value < WATCH:
        return "watch"
    return "action"


def cat_psi(ref: pd.Series, cur: pd.Series) -> float:
    """PSI for a categorical feature, over the union of observed categories."""
    ref_share = ref.value_counts(normalize=True)
    cur_share = cur.value_counts(normalize=True)
    total = 0.0
    for category in set(ref_share.index) | set(cur_share.index):
        expected = float(ref_share.get(category, EPSILON))
        actual = float(cur_share.get(category, EPSILON))
        total += (actual - expected) * np.log(actual / expected)
    return total


def report(
    path: str = "docs/drift_report.md",
    columns: tuple[str, ...] = ("int_rate", "dti", "annual_inc", "revol_util"),
) -> list[tuple[str, float, str]]:
    """Compare the validation window against the test window on numeric columns."""
    val = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")

    rows: list[tuple[str, float, str]] = []
    for column in columns:
        value = psi(val[column].dropna().to_numpy(), test[column].dropna().to_numpy())
        rows.append((column, value, band(value)))

    lines = ["# Drift report", "", "| feature | PSI | band |", "|---|---|---|"]
    lines += [f"| {name} | {value:.3f} | {label} |" for name, value, label in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    for name, value, label in report():
        print(f"{name} PSI {value:.3f} {label}")


if __name__ == "__main__":
    main()

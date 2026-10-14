"""Materialise the temporal train / validation / test splits from the raw archive.

Two properties are enforced as hard errors, not assertions:

* no block-listed (post-origination) column may enter the allow-list, and
* no leakage column may survive the ``usecols`` selection.

They raise rather than assert because Python removes ``assert`` statements
when it runs with ``-O``, which would turn this guard into a no-op in exactly
the optimised production image it is meant to protect.

Rows are additionally clipped to the same bounds the API validates against. A
field the serving schema rejects with 422 must not appear in the training set,
or the model silently learns a distribution it is never asked to score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.loan_default_risk.config import settings

RAW = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
OUT = Path("data/processed")

# Origination-time allow-list. See docs/leakage_audit.md for the per-column
# rationale on why every excluded column is post-decision.
ALLOW = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "revol_bal",
    "revol_util",
    "total_acc",
]

RESOLVED_STATUSES = ("Fully Paid", "Charged Off")

# Bounds mirroring src/api/schemas.py. Training and serving must agree, so a
# value the API would reject with 422 is clipped here rather than passed to the
# estimator.
FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "loan_amnt": (500, 40000),
    "int_rate": (5, 35),
    "installment": (50, 2000),
    "annual_inc": (10000, 500000),
    "dti": (0, 50),
    "fico_range_low": (300, 850),
    "fico_range_high": (300, 850),
    "revol_bal": (0, 100000),
    "revol_util": (0, 100),
}


def _reject_leakage(columns: list[str], source: str) -> None:
    """Raise if any block-listed column is present."""
    found = set(settings.leakage_blocklist).intersection(columns)
    if found:
        raise ValueError(f"leakage-prone column(s) {sorted(found)} present in {source}")


def clean_numeric(frame: pd.DataFrame) -> dict[str, int]:
    """Clip numeric fields to the bounds the API enforces.

    The raw corpus contains values the serving schema rejects: a DTI of -1 in
    the validation and test windows, and DTI above the 50 the API allows. A
    median-imputing preprocessor would not fix either, so they are clipped to
    the documented bounds and the counts are reported.

    Returns:
        Per-column count of rows clipped, for the data report.
    """
    clipped: dict[str, int] = {}
    for column, (low, high) in FIELD_BOUNDS.items():
        if column not in frame.columns:
            continue
        before = frame[column]
        # Values outside the bounds are recorded, then clipped into range.
        outside = int((~(before.between(low, high)) & before.notna()).sum())
        if outside:
            clipped[column] = outside
        frame[column] = before.clip(lower=low, upper=high)
    return clipped


def main() -> None:
    _reject_leakage(ALLOW, "the feature allow-list")

    if not RAW.is_file():
        raise FileNotFoundError(f"raw dataset not found at {RAW}; see data/README.md")

    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RAW, usecols=ALLOW, low_memory=False)
    _reject_leakage(list(frame.columns), "the raw dataset after column selection")

    resolved = frame[frame["loan_status"].isin(RESOLVED_STATUSES)].copy()
    resolved["target"] = (resolved["loan_status"] == "Charged Off").astype(int)
    resolved["issue_year"] = pd.to_datetime(resolved["issue_d"], format="%b-%Y").dt.year

    if resolved["issue_year"].isna().any():
        raise ValueError("unparseable issue_d values produced a null issue_year")

    # Clip before splitting so every window sees the same feature contract.
    clipped = clean_numeric(resolved)

    window_masks = {
        "train": resolved["issue_year"] <= settings.train_end,
        "val": resolved["issue_year"].between(settings.cal_year, settings.policy_year),
        "test": resolved["issue_year"] >= settings.test_start,
    }
    print(f"raw: {frame.shape} allow {len(ALLOW)} resolved {len(resolved)}")
    for column, count in sorted(clipped.items()):
        print(f"clipped {column} to the serving bounds: {count} rows")

    for name, mask in window_masks.items():
        subset = resolved[mask]
        subset.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name}: {subset.shape} dr={subset['target'].mean():.3f} max_year={int(subset['issue_year'].max())}")


if __name__ == "__main__":
    main()

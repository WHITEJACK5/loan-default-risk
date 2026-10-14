"""Materialise the temporal train / validation / test splits from the raw archive.

Two properties are enforced as hard errors, not assertions:

* no block-listed (post-origination) column may enter the allow-list, and
* no leakage column may survive the ``usecols`` selection.

They raise rather than assert because Python removes ``assert`` statements
when it runs with ``-O``, which would turn this guard into a no-op in exactly
the optimised production image it is meant to protect.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def _reject_leakage(columns: list[str], source: str) -> None:
    """Raise if any block-listed column is present."""
    found = settings.leakage_blocklist.intersection(columns)
    if found:
        raise ValueError(f"leakage-prone column(s) {sorted(found)} present in {source}")


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

    window_masks = {
        "train": resolved["issue_year"] <= settings.train_end,
        "val": resolved["issue_year"].between(settings.cal_year, settings.policy_year),
        "test": resolved["issue_year"] >= settings.test_start,
    }
    print(f"raw: {frame.shape} allow {len(ALLOW)} resolved {len(resolved)}")
    for name, mask in window_masks.items():
        subset = resolved[mask]
        subset.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name}: {subset.shape} dr={subset['target'].mean():.3f} year<={stats_year(subset)}")


def stats_year(subset: pd.DataFrame) -> int:
    """Highest issue year in a split, for a readable progress line."""
    return int(subset["issue_year"].max())


if __name__ == "__main__":
    main()

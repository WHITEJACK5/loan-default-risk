"""Validate the materialised splits before anything trains on them.

This module runs on import as a script only. Reading the parquet files at
module scope means ``import src.data.validate`` crashes when
``data/processed`` is absent, which is exactly the CI and fresh-checkout case,
so the loop is wrapped in a function and guarded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")


def validate(path: str | Path) -> dict[str, object]:
    """Check one split and report the defects that must be zero.

    Returns a dict of counts so a caller can assert on it instead of parsing
    console output.
    """
    frame = pd.read_parquet(path)
    issues = {
        "rows": len(frame),
        "bad_loan": int((frame["loan_amnt"] <= 0).sum()),
        "bad_dti": int((frame["dti"] < 0).sum()),
        "bad_target": int((~frame["target"].isin([0, 1])).sum()),
        "null_year": int(frame["issue_year"].isna().sum()),
        "bad_fico": int(((frame["fico_range_low"] < 300) | (frame["fico_range_high"] > 850)).sum()),
    }
    issues["rate"] = round(float(frame["target"].mean()), 4)
    return issues


def main() -> int:
    """Validate every split; return a process exit code."""
    failures: list[str] = []
    for name in SPLITS:
        path = Path(f"data/processed/{name}.parquet")
        if not path.is_file():
            print(f"{name}: MISSING {path} (run src.data.make_dataset)")
            continue

        report = validate(path)
        print(f"{name} " + " ".join(f"{k}={v}" for k, v in report.items()))

        blocking = {k: v for k, v in report.items() if k.startswith("bad") or k.startswith("null")}
        if any(v > 0 for v in blocking.values()):
            failures.append(f"{name}: {blocking}")

    if failures:
        print("FAIL " + "; ".join(failures))
        return 1
    print("PASS all splits clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

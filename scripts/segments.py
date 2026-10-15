"""Segment performance on the held-out test window.

This replaces a hand-written table whose numbers were not computed from the
data. Every value below comes from ``src/models/evaluate.py`` output on
``data/processed/test.parquet``; the approval rate is the share of each segment
that falls below the policy threshold.

This is a *descriptive* performance breakdown, not a fair-lending analysis: the
corpus has no protected attributes, and ``purpose`` and ``addr_state`` are
proxies for demographic mix at best.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.api.model import load_model
from src.loan_default_risk.config import settings

DROP_COLUMNS = ["target", "loan_status", "issue_d", "issue_year"]
SEGMENT_COLUMNS = ("purpose", "grade", "addr_state")


def segment_report(frame: pd.DataFrame, model) -> list[dict[str, object]]:
    """Score the test window and summarise each level of the segment columns."""
    features = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns])
    proba = model.predict_proba(features)[:, 1]

    rows: list[dict[str, object]] = []
    for column in SEGMENT_COLUMNS:
        for level, group in frame.groupby(column):
            mask = (frame[column] == level).to_numpy()
            if mask.sum() < 2:
                continue
            if group["target"].nunique() < 2:
                # AUC is undefined for a single-class group; PR-AUC still is.
                auc = None
            else:
                auc = round(float(roc_auc_score(group["target"], proba[mask])), 4)

            rows.append(
                {
                    "feature": column,
                    "level": str(level),
                    "n": int(mask.sum()),
                    "default_rate": round(float(group["target"].mean()), 4),
                    "mean_pd": round(float(proba[mask].mean()), 4),
                    "approval_rate": round(float((proba[mask] < settings.threshold).mean()), 4),
                    "roc": auc,
                }
            )
    return rows


def main() -> None:
    frame = pd.read_parquet("data/processed/test.parquet")
    model = load_model()
    rows = segment_report(frame, model)

    lines = [
        "# Segment performance (descriptive, not fair-lending)",
        "",
        f"Test window: {len(frame):,} loans, threshold {settings.threshold}.",
        "ROC is undefined for a single-class segment and is left blank.",
        "",
        "| feature | level | n | default rate | mean PD | approval rate | ROC |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['feature']} | {r['level']} | {r['n']:,} | {r['default_rate']:.3f} | "
        f"{r['mean_pd']:.3f} | {r['approval_rate']:.3f} | {r['roc'] if r['roc'] is not None else 'n/a'} |"
        for r in rows
    ]

    approval = pd.Series([r["approval_rate"] for r in rows])
    spread = round(float(approval.max() - approval.min()), 3)
    lines += ["", f"Approval-rate spread across all segments: {spread:.3f}", ""]

    Path("docs/segment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("docs/segment.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved docs/segment.md and docs/segment.json ({len(rows)} segments)")


if __name__ == "__main__":
    main()

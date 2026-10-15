"""Mean absolute SHAP contribution per source feature on the validation window.

``src/api/explain.py`` aggregates the one-hot block back to its source column,
so the ranking here is in the same terms the API returns in ``reasons``. The
figures are written to ``docs/drivers.md`` and are what the README cites.

Usage:
    python scripts/drivers.py [--sample 300]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.explain import credit_assignment
from src.api.model import load_model

DROP_COLUMNS = ["target", "loan_status", "issue_d", "issue_year"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=300, help="loans to average over")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_parquet("data/processed/val.parquet")
    features = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns])
    model = load_model()

    rng = np.random.default_rng(args.seed)
    index = rng.choice(len(features), size=min(args.sample, len(features)), replace=False)

    totals: dict[str, float] = {}
    for position in index:
        for feature, contribution in credit_assignment(model, features.iloc[[position]]).items():
            totals[feature] = totals.get(feature, 0.0) + abs(contribution)

    ranked = sorted(totals.items(), key=lambda item: -item[1])
    lines = [
        "# Top risk drivers",
        "",
        f"Mean |SHAP| per source feature over {len(index)} validation loans, aggregated by `src/api/explain.py`.",
        "",
        "| feature | mean contribution |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | {value / len(index):+.4f} |" for name, value in ranked]
    Path("docs/drivers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved docs/drivers.md ({len(ranked)} features)")
    for name, value in ranked[:6]:
        print(f"  {name:22} {value / len(index):+.4f}")


if __name__ == "__main__":
    main()

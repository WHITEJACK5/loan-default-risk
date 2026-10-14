"""Render the metrics block in README.md from artifacts/metrics.json.

``--check`` compares the rendered block against what is committed and exits
non-zero when they differ. That is what makes it usable as a CI gate: without
it the script always reported success, because "it ran" and "it found no
drift" looked identical in its output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

METRICS_FILE = Path("artifacts/metrics.json")
README_FILE = Path("README.md")
START = "<!-- METRICS:START -->"
END = "<!-- METRICS:END -->"

# Columns rendered per split, in display order. "dr" is the default rate.
COLUMNS = ["n", "dr", "ROC", "Gini", "PR-AUC", "Brier", "KS", "Recall_at_5pct"]
HEADER = ["split", *COLUMNS]


def render_block(metrics: dict) -> str:
    """Build the markdown table between the START and END markers."""
    header = "| " + " | ".join(HEADER) + " |"
    divider = "|" + "---|" * len(HEADER)
    rows = [header, divider]
    for split in ("train", "val", "test"):
        values = metrics.get(split)
        if values is None:
            continue
        cells = [split]
        for key in COLUMNS:
            value = values.get(key)
            cells.append("n/a" if value is None else _format(key, value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _format(key: str, value) -> str:
    """Integers as integers, floats to 4 significant places."""
    return str(int(value)) if key == "n" else f"{float(value):.4g}"


def replace_block(readme: str, block: str) -> str:
    """Swap the content between the markers, leaving everything else untouched."""
    if START not in readme or END not in readme:
        raise SystemExit(f"README.md is missing the {START} ... {END} markers")
    before, rest = readme.split(START, 1)
    _, after = rest.split(END, 1)
    return f"{before}{START}\n{block}\n{END}{after}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the README matches; exit 1 on drift")
    args = parser.parse_args()

    if not METRICS_FILE.is_file():
        print(f"error: {METRICS_FILE} not found; run src.models.train first", file=sys.stderr)
        return 2

    metrics = _read_json(METRICS_FILE)
    block = render_block(metrics)
    readme = README_FILE.read_text(encoding="utf-8")

    if args.check:
        if readme == replace_block(readme, block):
            print("in sync")
            return 0
        print("drift: README.md does not match artifacts/metrics.json", file=sys.stderr)
        return 1

    README_FILE.write_text(replace_block(readme, block), encoding="utf-8")
    print("rendered README metrics")
    return 0


def _read_json(path: Path) -> dict:
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())

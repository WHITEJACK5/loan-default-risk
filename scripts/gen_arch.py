"""Regenerate the architecture diagram in docs/architecture.png.

Labels are drawn from the same sources the README cites, so the diagram cannot
drift from the repository: the archive size is measured from the file on disk
and the model step names the calibration variant in artifacts/model_meta.json.
An earlier version hard-coded "392MB" and "Brier 0.155" and was stale the moment
either changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches

RAW_ARCHIVE = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
MODEL_META = Path("artifacts/model_meta.json")
OUTPUT = Path("docs/architecture.png")


def _archive_size() -> str:
    if not RAW_ARCHIVE.is_file():
        return "archive"
    return f"{RAW_ARCHIVE.stat().st_size / 1024 / 1024:.0f}MB"


def _calibration_label() -> str:
    if not MODEL_META.is_file():
        return "isotonic"
    meta = json.loads(MODEL_META.read_text())
    folds = meta.get("calibration_folds", 5)
    ensemble = meta.get("calibration_ensemble", False)
    return f"isotonic x{folds}" + ("" if ensemble else "\nsingle est")


def build() -> str:
    """Return the path written."""
    boxes = [
        f"data/raw\n{_archive_size()}",
        "make_dataset\ntemporal + clip",
        "build_features\nimpute + one-hot",
        "LightGBM\n" + _calibration_label(),
        "policy\nthr on val",
        "FastAPI\n/predict /explain",
        "MLflow + tests",
    ]

    figure, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2)
    ax.axis("off")

    x = 0.2
    for box in boxes:
        ax.add_patch(patches.Rectangle((x, 0.7), 1.4, 0.8, fill=False))
        ax.text(x + 0.7, 1.1, box, ha="center", va="center", fontsize=7)
        if x > 0.2:
            ax.arrow(x - 0.05, 1.1, 0.15, 0, head_width=0.08, head_length=0.08, fc="k")
        x += 1.65

    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=150)
    print(f"saved {OUTPUT} from the current archive and metadata")
    return str(OUTPUT)


if __name__ == "__main__":
    build()

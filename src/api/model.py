"""Model artifact loading with tamper-evidence.

A scikit-learn pipeline restored with joblib is executed pickle: loading a
model file is running code. Any process that accepts a model path from the
outside must therefore treat the file as untrusted input.

This module enforces two controls before the object is deserialized:

* the path must resolve inside an allow-listed base directory, so relative
  paths and ``..`` segments cannot escape it, and
* when ``artifacts/model_meta.json`` records ``sha256`` for the binary, the
  file must hash to exactly that value.

Both checks are cheap next to a predict call and turn "the model loaded" into
"the model loaded is the model we signed off".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

MODEL_PATH_ENV = "LDR_MODEL_PATH"
ARTIFACTS_DIR = Path(os.environ.get("LDR_ARTIFACTS_DIR", "artifacts"))
MODEL_METAFILE = ARTIFACTS_DIR / "model_meta.json"

_CHUNK = 1 << 20


class ModelIntegrityError(RuntimeError):
    """Raised when the model binary fails its path or digest check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_model_path(path: str | Path) -> Path:
    """Constrain the model path to the artifacts directory or an env override."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (ARTIFACTS_DIR / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not str(candidate).startswith(str(ARTIFACTS_DIR.resolve())):
        raise ModelIntegrityError(
            f"model path {candidate} escapes the artifacts directory {ARTIFACTS_DIR.resolve()}"
        )
    return candidate


def load_model(path: str | Path | None = None):
    """Load the persisted model, refusing an altered or out-of-bounds binary.

    Args:
        path: Model binary. Defaults to the ``LDR_MODEL_PATH`` environment
            variable, then ``artifacts/model.joblib``.

    Raises:
        FileNotFoundError: No model binary at the resolved path.
        ModelIntegrityError: Path escapes the artifacts directory, or the file
            does not match the SHA-256 recorded in ``model_meta.json``.
    """
    target = _resolve_model_path(path or os.environ.get(MODEL_PATH_ENV, "model.joblib"))

    if not target.is_file():
        raise FileNotFoundError(
            f"model artifact not found at {target}. Train first with `python -m src.models.train` "
            f"or set {MODEL_PATH_ENV}."
        )

    expected = None
    if MODEL_METAFILE.is_file():
        meta = json.loads(MODEL_METAFILE.read_text(encoding="utf-8"))
        expected = meta.get("model_sha256")
    if expected:
        actual = _sha256(target)
        if actual != expected.lower():
            raise ModelIntegrityError(
                f"model integrity check failed for {target}: expected sha256 {expected}, got {actual}"
            )
        logger.info("model integrity verified sha256=%s", expected)
    else:
        logger.warning("no sha256 recorded in %s; loaded without integrity check", MODEL_METAFILE)

    return joblib.load(target)

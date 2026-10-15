"""Model artifact loading with tamper-evidence.

A scikit-learn pipeline restored with joblib is executed pickle: loading a
model file is running code. Any process that accepts a model path from the
outside must therefore treat the file as untrusted input.

This module enforces two controls before the object is deserialized:

* the path must resolve inside the allow-listed artifacts directory, so
  relative paths and ``..`` segments cannot escape it, and
* when ``model_meta.json`` records ``sha256`` for the binary, the file must
  hash to exactly that value.

Paths are resolved per call rather than frozen at import, so the same process
can be pointed at a different artifact directory without a re-import. That is
what lets the test suite and the production image share one code path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

MODEL_PATH_ENV = "LDR_MODEL_PATH"
ARTIFACTS_DIR_ENV = "LDR_ARTIFACTS_DIR"
META_FILENAME = "model_meta.json"

_CHUNK = 1 << 20


class ModelIntegrityError(RuntimeError):
    """Raised when the model binary fails its path or digest check."""


def artifacts_dir() -> Path:
    """Resolved artifacts directory, read from the environment per call."""
    return Path(os.environ.get(ARTIFACTS_DIR_ENV, "artifacts")).resolve()


def meta_path() -> Path:
    return artifacts_dir() / META_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_path(path: str | Path | None = None) -> Path:
    """Resolve a model path inside the artifacts directory.

    Args:
        path: Model binary, absolute or relative to the artifacts directory.
            Defaults to ``LDR_MODEL_PATH``, then ``model.joblib``.

    Raises:
        ModelIntegrityError: The resolved path escapes the artifacts directory.
    """
    base = artifacts_dir()
    candidate = Path(path or os.environ.get(MODEL_PATH_ENV, "model.joblib")).expanduser()
    candidate = candidate if candidate.is_absolute() else base / candidate
    candidate = candidate.resolve()

    if candidate != base and base not in candidate.parents:
        raise ModelIntegrityError(f"model path {candidate} escapes the artifacts directory {base}")
    return candidate


def load_model(path: str | Path | None = None) -> CalibratedClassifierCV:
    """Load the persisted model, refusing an altered or out-of-bounds binary.

    Args:
        path: Model binary. Defaults to ``LDR_MODEL_PATH`` then
            ``artifacts/model.joblib``.

    Raises:
        FileNotFoundError: No model binary at the resolved path.
        ModelIntegrityError: The path escapes the artifacts directory, or the
            file does not match the SHA-256 recorded in ``model_meta.json``.
    """
    target = resolve_model_path(path)

    if not target.is_file():
        raise FileNotFoundError(
            f"model artifact not found at {target}. Train first with `python -m src.models.train` "
            f"or set {MODEL_PATH_ENV}."
        )

    manifest = meta_path()
    expected = json.loads(manifest.read_text(encoding="utf-8")).get("model_sha256") if manifest.is_file() else None

    if expected:
        actual = _sha256(target)
        if actual != str(expected).lower():
            raise ModelIntegrityError(
                f"model integrity check failed for {target}: expected sha256 {expected}, got {actual}"
            )
        logger.info("model integrity verified sha256=%s", expected)
    else:
        logger.warning("no sha256 recorded in %s; loaded without integrity check", manifest)

    return joblib.load(target)

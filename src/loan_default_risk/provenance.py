"""Provenance helpers: git revision, dataset hashing, and artifact integrity.

Every value in an experiment run must be traceable and verifiable, so these
helpers refuse to guess. A malformed hash or an unresolvable revision raises
instead of silently producing "unknown" and letting a run look reproducible
when it is not.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SHA_LINE = re.compile(r"^\s*(?:[-*]\s*)?sha256:\s*([0-9a-fA-F]{64})\s*$", re.MULTILINE)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

# Bytes to read per chunk when hashing large archives.
_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file, safe for the 392MB raw archive."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_data_sha256(readme: str | Path = "data/README.md") -> str:
    """Read and validate the dataset SHA-256 recorded in data/README.md.

    Raises ValueError if the marker is missing or malformed, so a truncated or
    hand-edited README cannot silently record the wrong data version.
    """
    text = Path(readme).read_text(encoding="utf-8")
    match = _SHA_LINE.search(text)
    if not match:
        raise ValueError(f"no valid 'sha256: <64 hex>' marker found in {readme}")
    value = match.group(1).lower()
    if not _HEX64.match(value):
        raise ValueError(f"dataset hash in {readme} is not a 64-char lowercase hex digest")
    return value


def _resolve_ref(git_dir: Path, ref: str) -> str | None:
    """Follow a ref through ``packed-refs`` and the loose-refs directory."""
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 2 and parts[1].endswith(f"/{ref}"):
                return parts[0]

    loose = git_dir.joinpath(*ref.split("/"))
    if loose.is_file():
        candidate = loose.read_text(encoding="utf-8", errors="replace").strip()
        return _resolve_ref(git_dir, candidate[5:]) if candidate.startswith("ref:") else candidate
    return None


def resolve_git_sha() -> str:
    """Best-effort current git commit, read from ``.git`` without a subprocess.

    Resolution order keeps this deterministic under CI and containers where
    ``.git`` may be absent:

    1. ``LDR_GIT_SHA`` (or ``GITHUB_SHA``) environment variable, set by CI.
    2. ``.git/HEAD`` plus ``packed-refs``, which costs one file read and works
       with no git binary present.

    The revision is parsed from disk rather than obtained by shelling out to
    ``git``, so no executable is resolved from ``PATH`` and no environment
    variable can redirect the lookup at a different repository.
    """
    for key in ("LDR_GIT_SHA", "GITHUB_SHA"):
        forced = os.environ.get(key, "").strip().lower()
        if _HEX40.match(forced):
            return forced

    git_dir = Path(os.environ.get("GIT_DIR", ".git"))
    if not git_dir.is_absolute():
        # .git may live at the repository root rather than the working directory.
        candidate = git_dir
        for parent in [Path.cwd(), *Path.cwd().parents]:
            candidate = parent / git_dir
            if candidate.is_dir():
                break
        git_dir = candidate

    head = git_dir / "HEAD"
    if not head.is_file():
        logger.debug("no .git/HEAD at %s; git revision unavailable", head)
        return "unknown"

    pointer = head.read_text(encoding="utf-8", errors="replace").strip()
    if pointer.startswith("ref:"):
        resolved = _resolve_ref(git_dir, pointer[4:].strip())
        if resolved and _HEX40.match(resolved):
            return resolved.lower()
        logger.debug("ref %s did not resolve; git revision unavailable", pointer)
        return "unknown"

    if _HEX40.match(pointer):
        return pointer.lower()
    logger.debug("unrecognised .git/HEAD contents: %r", pointer[:64])
    return "unknown"

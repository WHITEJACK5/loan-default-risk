"""Leakage guards on the feature allow-list and the model path.

The check that survives today's refactor is the point: ``make_dataset`` asserts
nothing about columns that leak the label, and the model loader refuses a path
outside its artifacts directory. Both are the controls that stop a post-decision
column from entering the feature set or a foreign pickle from being executed.
"""

from __future__ import annotations

from src.loan_default_risk.config import settings

BLOCKLIST = set(settings.leakage_blocklist)


def test_no_leakage_columns_in_allow_list():
    from src.data.make_dataset import ALLOW

    leaked = BLOCKLIST.intersection(ALLOW)
    assert not leaked, f"post-origination columns in the feature allow-list: {sorted(leaked)}"


def test_allow_list_is_origination_time_features():
    from src.data.make_dataset import ALLOW

    # Columns whose meaning is only knowable at or after funding. If any of
    # these appear in ALLOW, the test suite has a leak it can detect statically.
    post_decision = {
        "out_prncp",
        "total_pymnt",
        "recoveries",
        "last_pymnt_d",
        "last_credit_pull_d",
        "hardship_flag",
        "settlement_status",
        "collection_recovery_fee",
        "grade_later",
    }
    leaked = post_decision.intersection(ALLOW)
    assert not leaked, f"post-decision columns leaked into ALLOW: {sorted(leaked)}"


def _features_for(lender_side: str) -> set[str]:
    """Columns the preprocessor is configured to consume, unfitted.

    ``.transformers`` is the constructor argument and is always present;
    ``.transformers_`` only exists after ``fit``, so asserting on it here
    would require training a model just to check a column list.
    """
    from src.features.build_features import get_preprocessor

    preprocessor = get_preprocessor(lender_side=lender_side)
    return {column for _, _, columns in preprocessor.transformers for column in columns}


def test_metadata_columns_are_not_model_features():
    """The pipeline must not see target, loan_status, issue_d or issue_year."""
    drop_columns = {"target", "loan_status", "issue_d", "issue_year"}
    for side in ("A", "B"):
        leaked = drop_columns.intersection(_features_for(side))
        assert not leaked, f"lender-{side} preprocessor sees metadata columns: {sorted(leaked)}"


def test_lender_side_b_drops_disclosure_grade_features():
    """Lender B removes features used for pricing: grade, sub_grade, int_rate, installment."""
    side_b = _features_for("B")
    assert {"grade", "sub_grade", "int_rate", "installment"}.isdisjoint(side_b)


def test_lender_side_a_keeps_the_disclosure_features():
    side_a = _features_for("A")
    assert {"grade", "int_rate"}.issubset(side_a), "lender A is the pricing-informed view"


def test_blocklisted_columns_are_never_configured():
    from src.data.make_dataset import ALLOW

    leaked = BLOCKLIST.intersection(set(ALLOW) | _features_for("A") | _features_for("B"))
    assert not leaked, f"block-listed columns reach the feature set: {sorted(leaked)}"


def test_model_loader_rejects_paths_outside_artifacts(tmp_path, monkeypatch):
    """A path traversal must not let a foreign pickle be loaded."""
    from src.api.model import ModelIntegrityError, resolve_model_path

    monkeypatch.setenv("LDR_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("LDR_MODEL_PATH", "../../etc/passwd")
    try:
        resolve_model_path()
    except ModelIntegrityError:
        return
    raise AssertionError("expected ModelIntegrityError for a traversal path")


def test_model_loader_rejects_digest_mismatch(tmp_path, monkeypatch):
    """A tampered binary must be refused before it is deserialized."""
    import json

    from src.api.model import ModelIntegrityError, load_model

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model.joblib").write_bytes(b"not-a-pickle")
    (artifacts / "model_meta.json").write_text(json.dumps({"model_sha256": "0" * 64}))

    monkeypatch.setenv("LDR_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("LDR_MODEL_PATH", str(artifacts / "model.joblib"))

    try:
        load_model()
    except ModelIntegrityError:
        return
    raise AssertionError("expected ModelIntegrityError for a digest mismatch")

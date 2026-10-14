"""Shared test fixtures.

The API loads a model from disk, so a test run needs a model on disk. Fitting
a 5-fold isotonic LightGBM on 451k rows takes minutes and would make the suite
unusable, so the default fixture is a small calibrated model trained on the
seeded synthetic corpus in ``tests/fixtures/synthetic.parquet``.

Because ``src.api.model`` resolves ``LDR_ARTIFACTS_DIR`` and ``LDR_MODEL_PATH``
at call time rather than at import time, ``monkeypatch`` is enough to redirect
the loader; no module reload is required, and the production code and the test
code follow the same path.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from src.loan_default_risk.modeling import build_model

FIXTURE_DIR = "tests/fixtures"
DROP_COLUMNS = ["target", "loan_status", "issue_d", "issue_year"]

# Fixture-model hyper-parameters: enough signal to rank, few enough trees to
# keep the suite at seconds. The tests assert on behaviour, not coefficients.
FIXTURE_TREES = 40
FIXTURE_FOLDS = 2


def wants_real_model() -> bool:
    return os.environ.get("LDR_TEST_REAL_MODEL", "").lower() in {"1", "true", "yes"}


@pytest.fixture(scope="session")
def synthetic_split() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Train and held-out halves of the seeded synthetic corpus."""
    frame = pd.read_parquet(f"{FIXTURE_DIR}/synthetic.parquet")
    features = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns])
    cut = len(frame) // 2
    return {
        "train": (features.iloc[:cut], frame["target"].iloc[:cut]),
        "test": (features.iloc[cut:], frame["target"].iloc[cut:]),
    }


@pytest.fixture(scope="session")
def fixture_model(synthetic_split, tmp_path_factory) -> str:
    """Path to a fresh calibrated model, trained once per session.

    The artifact is deliberately *not* written to ``artifacts/``, so it cannot
    be mistaken for a real model or picked up by other tooling.
    """
    x_train, y_train = synthetic_split["train"]
    x_test, y_test = synthetic_split["test"]

    base = build_model(lender_side="B")
    base.set_params(clf__n_estimators=FIXTURE_TREES, clf__learning_rate=0.1)
    calibrated = CalibratedClassifierCV(estimator=base, method="isotonic", cv=FIXTURE_FOLDS)
    calibrated.fit(x_train, y_train)

    p = calibrated.predict_proba(x_test)[:, 1]
    if not np.isfinite(p).all():
        raise AssertionError("fixture model produced non-finite probabilities")
    if not (p.min() >= 0.0 and p.max() <= 1.0):
        raise AssertionError(f"fixture PD out of range: [{p.min()}, {p.max()}]")

    auc = roc_auc_score(y_test, p)
    if auc <= 0.55:
        raise AssertionError(f"fixture model does not rank above chance (AUC {auc:.3f})")

    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(calibrated, path)
    return str(path)


@pytest.fixture()
def api(fixture_model, monkeypatch):
    """TestClient for the API, wired to the session's fixture model.

    Set ``LDR_TEST_REAL_MODEL=1`` before the run to exercise the real artifact
    instead; that requires ``data/processed`` and ``artifacts/model.joblib``.
    """
    if wants_real_model():
        artifacts = Path("artifacts")
        model = artifacts / "model.joblib"
    else:
        artifacts = Path(fixture_model).parent
        model = Path(fixture_model)

    monkeypatch.setenv("LDR_ARTIFACTS_DIR", artifacts)
    monkeypatch.setenv("LDR_MODEL_PATH", model)

    from fastapi.testclient import TestClient

    import src.api.main as api_main

    with TestClient(api_main.app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture()
def sample_payload() -> dict:
    """A valid applicant, within every bound in ``src/api/schemas.py``."""
    return {
        "loan_amnt": 10000,
        "term": "36 months",
        "int_rate": 13.5,
        "installment": 300,
        "grade": "C",
        "sub_grade": "C3",
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "annual_inc": 65000,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "dti": 18,
        "delinq_2yrs": 0,
        "fico_range_low": 680,
        "fico_range_high": 684,
        "inq_last_6mths": 1,
        "open_acc": 10,
        "revol_bal": 15000,
        "revol_util": 45,
        "total_acc": 20,
    }


@pytest.fixture()
def risky_payload(sample_payload) -> dict:
    """An applicant that must score a higher PD than ``sample_payload``."""
    return {
        **sample_payload,
        "dti": 35,
        "annual_inc": 20000,
        "fico_range_low": 600,
        "fico_range_high": 604,
        "revol_util": 95,
    }

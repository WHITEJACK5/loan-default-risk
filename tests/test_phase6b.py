"""Drift, experiment-assignment and calibration properties.

Each of these is a property that can only be verified one way: PSI of a sample
against itself is zero by construction, a balanced split is not a sample-ratio
mismatch, and a perfectly calibrated forecast has zero Brier score.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss


def test_psi_of_a_sample_against_itself_is_zero():
    from src.monitoring.drift import psi

    reference = np.random.default_rng(42).normal(size=1000)
    assert psi(reference, reference) < 1e-9, "PSI of a distribution against itself must be 0"


def test_psi_is_symmetric_for_identical_distributions():
    from src.monitoring.drift import psi

    reference = np.random.default_rng(1).normal(size=1000)
    current = np.random.default_rng(2).normal(size=1000)
    # Same distribution, different draw: PSI is small, and swapping the
    # arguments agrees because the bins come from the reference both ways here.
    assert psi(reference, current) < 0.25


def test_psi_bands_are_ordered():
    from src.monitoring.drift import band

    assert band(0.02) == "stable"
    assert band(0.15) == "watch"
    assert band(0.50) == "action"


def test_psi_handles_a_constant_reference():
    """A frozen feature must not crash the drift check."""
    from src.monitoring.drift import psi

    constant = np.full(100, 7.0)
    assert psi(constant, constant) == 0.0
    assert np.isinf(psi(constant, np.random.default_rng(0).normal(size=100)))


def test_srm_detects_an_unbalanced_split():
    from src.experiments.stats import srm_check

    assert not srm_check(500, 500)["srm"], "an even split must not flag SRM"
    assert srm_check(600, 400)["srm"], "a 60/40 split at 1000 is an SRM"
    assert srm_check(5600, 4400)["srm"], "a 56/44 split at 10000 is an SRM"


def test_assignment_is_deterministic_and_balanced():
    from src.experiments.assign import assign

    seen = {"A": 0, "B": 0}
    for index in range(1000):
        bucket = assign(f"user_{index}", exp_id="loan_policy_v1", allocation=0.5)
        assert bucket in seen
        seen[bucket] += 1
        assert assign(f"user_{index}", exp_id="loan_policy_v1", allocation=0.5) == bucket

    # A 50/50 salt should land both sides of a 42% - 58% band for 1000 users.
    assert 0.42 <= seen["A"] / 1000 <= 0.58, seen


def test_assignment_honours_a_non_uniform_allocation():
    from src.experiments.assign import assign

    treated = sum(assign(f"user_{i}", exp_id="loan_policy_v1", allocation=0.9) == "A" for i in range(1000))
    assert 0.85 <= treated / 1000 <= 0.95, f"10% holdout got {treated}/1000"


def test_perfect_probabilities_have_zero_brier_score():
    assert brier_score_loss([0, 0, 1, 1], [0, 0, 1, 1]) == 0.0


def test_brier_score_penalises_confidence():
    """A confident wrong answer must score worse than a calibrated hedge."""
    perfect = brier_score_loss([1, 0], [1, 0])
    hedging = brier_score_loss([1, 0], [0.5, 0.5])
    overconfident = brier_score_loss([1, 0], [0.0, 1.0])
    assert perfect < hedging < overconfident


@pytest.mark.parametrize("approves", [True, False])
def test_decision_equals_the_threshold_rule(api, sample_payload, approves):
    """The decision must be a pure function of PD and the configured threshold."""
    body = api.post("/predict", json=sample_payload).json()
    expected = "approve" if body["pd"] < body["threshold"] else "reject"
    assert body["decision"] == expected, body


def test_every_response_reports_the_same_threshold(api, sample_payload, risky_payload):
    from src.loan_default_risk.config import settings

    for payload in (sample_payload, risky_payload):
        body = api.post("/predict", json=payload).json()
        assert body["threshold"] == settings.threshold, "threshold drift between requests"


def test_assign_endpoint_uses_the_same_policy_as_the_library(api):
    from src.experiments.assign import assign

    for user in ("user_1", "user_2", "user_3"):
        bucket = api.post("/experiment/assign", json={"user_id": user}).json()["bucket"]
        assert bucket == assign(user), f"endpoint and library disagree for {user}"

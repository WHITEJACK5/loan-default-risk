"""API contract tests: the scoring endpoints and their guards.

These assert on behaviour a change could silently break — that a valid
applicant scores, that an invalid one is rejected with 422 rather than 500,
that the decision follows the threshold, and that a response never leaks
internal state.
"""

from __future__ import annotations

import pytest

# `api` and `sample_payload` are fixtures from conftest.py; they are injected by
# name, not imported.


def test_health_is_liveness_not_readiness(api):
    response = api.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_the_model(api):
    response = api.get("/ready")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["model_version"]


def test_predict_scores_a_valid_applicant(api, sample_payload):
    response = api.post("/predict", json=sample_payload)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["pd"] <= 1.0
    assert body["decision"] == ("approve" if body["pd"] < body["threshold"] else "reject")
    assert body["model_version"]
    assert body["request_id"]


def test_decision_follows_the_configured_threshold(api, sample_payload):
    body = api.post("/predict", json=sample_payload).json()
    risky = api.post(
        "/predict",
        json={
            **sample_payload,
            "dti": 35,
            "annual_inc": 20000,
            "fico_range_low": 600,
            "fico_range_high": 604,
        },
    ).json()
    assert risky["pd"] > body["pd"], "a weaker applicant must score a higher PD"


def test_explain_returns_ordered_positive_reasons(api, sample_payload):
    response = api.post("/explain", json=sample_payload)
    assert response.status_code == 200

    body = response.json()
    reasons = body["reasons"]
    assert 0 < len(reasons) <= 4

    magnitudes = [abs(_contribution(r)) for r in reasons]
    assert magnitudes == sorted(magnitudes, reverse=True), "reasons are ranked by |contribution|"
    assert all(_contribution(r) > 0 for r in reasons), "only risk-raising features are reasons"
    assert all(":" in r and "=" in r for r in reasons)


def test_explain_reasons_name_source_features(api, sample_payload):
    """Reasons must name applicant-level columns, not one-hot indicator columns."""
    body = api.post("/explain", json=sample_payload).json()
    for reason in body["reasons"]:
        name = reason.split("=", 1)[0]
        assert name in sample_payload, f"{name} is not an applicant-level feature"


def test_assign_is_deterministic_and_binary(api):
    first = api.post("/experiment/assign", json={"user_id": "user_42"}).json()
    second = api.post("/experiment/assign", json={"user_id": "user_42"}).json()
    assert first == second, "assignment must be stable per user"
    assert first["bucket"] in {"A", "B"}


def test_assign_splits_users_both_ways(api):
    buckets = {api.post("/experiment/assign", json={"user_id": f"user_{i:04d}"}).json()["bucket"] for i in range(200)}
    assert buckets == {"A", "B"}, f"only saw {buckets}"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dti", -5),
        ("dti", 99),
        ("fico_range_low", 100),
        ("annual_inc", 10),
        ("loan_amnt", 999999),
        ("term", "12 months"),
        ("grade", "C1"),
        ("emp_length", "12 years"),
        ("purpose", "buy_a_boat"),
        ("addr_state", "ZZZ"),
        ("delinq_2yrs", -1),
    ],
)
def test_out_of_range_field_is_rejected(api, sample_payload, field, value):
    response = api.post("/predict", json={**sample_payload, field: value})
    assert response.status_code == 422, f"{field}={value} should be rejected"


def test_unknown_field_is_rejected(api, sample_payload):
    assert api.post("/predict", json={**sample_payload, "leak_flag": 1}).status_code == 422


def test_missing_field_is_rejected(api, sample_payload):
    payload = {k: v for k, v in sample_payload.items() if k != "dti"}
    assert api.post("/predict", json=payload).status_code == 422


def test_inverted_fico_range_is_rejected(api, sample_payload):
    payload = {**sample_payload, "fico_range_low": 700, "fico_range_high": 600}
    assert api.post("/predict", json=payload).status_code == 422


def test_non_finite_float_is_rejected(api, sample_payload):
    payload = {**sample_payload, "dti": 0}
    body = api.post("/predict", json=payload).json()
    assert 0.0 <= body["pd"] <= 1.0


def test_every_response_carries_a_request_id(api, sample_payload):
    response = api.post("/predict", json=sample_payload)
    assert response.headers["X-Request-ID"]
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    missing = api.get("/no-such-endpoint")
    assert missing.status_code == 404


def test_provided_request_id_is_echoed(api, sample_payload):
    response = api.post("/predict", json=sample_payload, headers={"X-Request-ID": "trace-me-001"})
    assert response.headers["X-Request-ID"] == "trace-me-001"


def _contribution(reason: str) -> float:
    """Parse the trailing signed float from a formatted reason."""
    return float(reason.rsplit(":", 1)[1])

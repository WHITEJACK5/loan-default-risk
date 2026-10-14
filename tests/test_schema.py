"""Input-schema guards: what the API rejects, and how it says why.

A 422 means the request never reached the model. A 500 means it did and the
model failed. Conflating the two turns a caught validation error into an
opaque server error, which is the bug the schema validation exists to prevent.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        ("dti", -5, "negative DTI is impossible"),
        ("dti", 99.9, "DTI above the corpus maximum"),
        ("fico_range_low", 100, "FICO below the scoring floor"),
        ("fico_range_high", 9999, "FICO above the scoring ceiling"),
        ("annual_inc", 10, "income below the corpus minimum"),
        ("loan_amnt", 999999, "principal above the corpus maximum"),
        ("int_rate", 100, "rate above the corpus maximum"),
        ("revol_util", 101, "utilisation above 100 percent"),
        ("delinq_2yrs", -1, "count cannot be negative"),
        ("open_acc", 0, "an applicant has at least one open account"),
    ],
)
def test_out_of_range_field_is_rejected(api, sample_payload, field, value, why):
    response = api.post("/predict", json={**sample_payload, field: value})
    assert response.status_code == 422, f"{field}={value}: {why}"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("term", "12 months"),
        ("grade", "C1"),
        ("emp_length", "12 years"),
        ("home_ownership", "CASTLE"),
        ("verification_status", "Maybe"),
        ("purpose", "buy_a_boat"),
        ("addr_state", "ZZZ"),
        ("addr_state", "CA1"),
        ("sub_grade", "H1"),
    ],
)
def test_unknown_category_is_rejected(api, sample_payload, field, value):
    assert api.post("/predict", json={**sample_payload, field: value}).status_code == 422


def test_missing_required_field_is_rejected(api, sample_payload):
    payload = {k: v for k, v in sample_payload.items() if k != "dti"}
    assert api.post("/predict", json=payload).status_code == 422


def test_unknown_field_is_rejected(api, sample_payload):
    """Post-origination fields must not be silently accepted and ignored."""
    payload = {**sample_payload, "total_pymnt": 12345, "recoveries": 99}
    assert api.post("/predict", json=payload).status_code == 422


def test_inverted_fico_range_is_rejected(api, sample_payload):
    payload = {**sample_payload, "fico_range_low": 700, "fico_range_high": 600}
    assert api.post("/predict", json=payload).status_code == 422


def test_error_body_identifies_the_offending_field(api, sample_payload):
    """The 422 must name the field, not just say "validation error"."""
    response = api.post("/predict", json={**sample_payload, "dti": -5})
    locations = [".".join(str(part) for part in error["loc"]) for error in response.json()["detail"]]
    assert any("dti" in location for location in locations), locations


def test_valid_boundary_values_are_accepted(api, sample_payload):
    payload = {
        **sample_payload,
        "loan_amnt": 500,
        "int_rate": 5.0,
        "annual_inc": 10000,
        "dti": 0,
        "revol_util": 0,
        "fico_range_low": 300,
        "fico_range_high": 300,
        "open_acc": 1,
        "total_acc": 1,
        "delinq_2yrs": 10,
    }
    response = api.post("/predict", json=payload)
    assert response.status_code == 200, "boundary values within bounds must score"

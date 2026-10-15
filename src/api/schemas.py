"""Request and response contracts.

Two rules shape this module:

1. **The schema must be at least as wide as the model's training distribution.**
   An earlier version allowed 4 loan purposes while the pipeline is trained on
   14, so 10 of them returned 422 and could not be scored. A schema narrower
   than the model is a functional bug that presents as bad input.
2. **Bounds must match the bounds the training data was clipped to.** Values the
   API rejects are clipped away in ``src/data/make_dataset.py``; a request
   outside them cannot be scored by a model that never saw such a value.

A ``LoanRequest`` is usable without FastAPI: ``to_frame()`` builds the
single-row ``DataFrame`` the pipeline expects, so the API, the batch policy
sweep and the dashboard share one construction path.
"""

from __future__ import annotations

import math

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Category sets as they appear in the training corpus. These are the values the
# one-hot encoder was fitted on, not a curated subset. Keeping them in sync with
# src/data/make_dataset.ALLOW is what stops the silent all-zero-vector bug that
# OneHotEncoder(handle_unknown="ignore") produces for an unpadded category.
TERMS = ["36 months", "60 months"]
GRADES = list("ABCDEFG")
EMP_LENGTHS = [
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
]
HOME_OWNERSHIP = ["RENT", "MORTGAGE", "OWN", "ANY", "NONE", "OTHER"]
VERIFICATION = ["Verified", "Source Verified", "Not Verified"]
PURPOSES = [
    "car",
    "credit_card",
    "debt_consolidation",
    "educational",
    "home_improvement",
    "house",
    "major_purchase",
    "medical",
    "moving",
    "other",
    "renewable_energy",
    "small_business",
    "vacation",
    "wedding",
]

# Bounds mirror FIELD_BOUNDS in src/data/make_dataset.py, which clip the
# training corpus. Train and serve must agree or the model is asked to score
# values it never saw.
FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "loan_amnt": (500.0, 40000.0),
    "int_rate": (5.0, 35.0),
    "installment": (50.0, 2000.0),
    "annual_inc": (10000.0, 500000.0),
    "dti": (0.0, 50.0),
    "revol_bal": (0.0, 100000.0),
    "revol_util": (0.0, 100.0),
}
INTEGER_BOUNDS: dict[str, tuple[int, int]] = {
    "fico_range_low": (300, 850),
    "fico_range_high": (300, 850),
    "delinq_2yrs": (0, 10),
    "inq_last_6mths": (0, 10),
    "open_acc": (1, 50),
    "total_acc": (1, 100),
}
CHOICE_FIELDS = {
    "term": TERMS,
    "grade": GRADES,
    "emp_length": EMP_LENGTHS,
    "home_ownership": HOME_OWNERSHIP,
    "verification_status": VERIFICATION,
    "purpose": PURPOSES,
}


class LoanRequest(BaseModel):
    """One loan application in the origination-time schema."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    loan_amnt: float = Field(ge=FIELD_BOUNDS["loan_amnt"][0], le=FIELD_BOUNDS["loan_amnt"][1])
    term: str
    int_rate: float = Field(ge=FIELD_BOUNDS["int_rate"][0], le=FIELD_BOUNDS["int_rate"][1])
    installment: float = Field(ge=FIELD_BOUNDS["installment"][0], le=FIELD_BOUNDS["installment"][1])
    grade: str
    sub_grade: str = Field(pattern="^[A-G][1-5]$")
    emp_length: str
    home_ownership: str
    annual_inc: float = Field(ge=FIELD_BOUNDS["annual_inc"][0], le=FIELD_BOUNDS["annual_inc"][1])
    verification_status: str
    purpose: str
    addr_state: str = Field(pattern="^[A-Za-z]{2}$", description="US state abbreviation")
    dti: float = Field(ge=FIELD_BOUNDS["dti"][0], le=FIELD_BOUNDS["dti"][1])
    delinq_2yrs: int
    fico_range_low: int
    fico_range_high: int
    inq_last_6mths: int
    open_acc: int
    revol_bal: float = Field(ge=FIELD_BOUNDS["revol_bal"][0], le=FIELD_BOUNDS["revol_bal"][1])
    revol_util: float = Field(ge=FIELD_BOUNDS["revol_util"][0], le=FIELD_BOUNDS["revol_util"][1])
    total_acc: int

    @field_validator(*CHOICE_FIELDS)
    @classmethod
    def _in_category(cls, value: str, info) -> str:
        allowed = CHOICE_FIELDS.get(info.field_name)
        if allowed is None:
            return value
        if value not in allowed:
            raise ValueError(f"{info.field_name} must be one of {allowed}")
        return value

    @field_validator("addr_state")
    @classmethod
    def _upper_state(cls, value: str) -> str:
        """States are upper-case in the training corpus; accept either casing."""
        return value.upper()

    @field_validator("sub_grade")
    @classmethod
    def _sub_grade_matches_grade(cls, value: str, info) -> str:
        grade = info.data.get("grade")
        if grade is not None and value and not value.startswith(grade):
            raise ValueError(f"sub_grade {value} must belong to grade {grade}")
        return value.upper()

    @field_validator("fico_range_high")
    @classmethod
    def _fico_ordered(cls, value: int, info) -> int:
        low = info.data.get("fico_range_low")
        if low is not None and value < low:
            raise ValueError("fico_range_high must be >= fico_range_low")
        return value

    @field_validator(*FIELD_BOUNDS)
    @classmethod
    def _finite(cls, value: float) -> float:
        """NaN and infinity pass ge/le only because they compare False."""
        if not math.isfinite(value):
            raise ValueError("value must be finite (NaN and infinity are rejected)")
        return value

    @field_validator(*INTEGER_BOUNDS)
    @classmethod
    def _bounded_int(cls, value: int, info) -> int:
        low, high = INTEGER_BOUNDS[info.field_name]
        if not low <= value <= high:
            raise ValueError(f"{info.field_name} must be between {low} and {high}, got {value}")
        return value

    def to_frame(self) -> pd.DataFrame:
        """Convert to the single-row frame the scoring pipeline consumes."""
        return pd.DataFrame([self.model_dump()])


def _make_response_base() -> type[BaseModel]:
    return BaseModel


class PredictResponse(BaseModel):
    """Score and the policy decision derived from it."""

    pd: float = Field(ge=0.0, le=1.0, description="calibrated probability of default")
    decision: str = Field(pattern="^(approve|reject)$")
    threshold: float
    model_version: str
    request_id: str


class ExplainResponse(PredictResponse):
    """Score plus the features that contributed to it."""

    reasons: list[str] = Field(default_factory=list, max_length=8)


class AssignRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    exp_id: str = Field(default="loan_policy_v1", min_length=1, max_length=128)


class AssignResponse(BaseModel):
    bucket: str = Field(pattern="^[AB]$")

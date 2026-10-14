"""Request and response contracts.

``LoanRequest`` is deliberately strict: unknown keys are rejected and every
field is bounded. The estimator only sees imputed, one-hot vectors, so tight
input bounds are the only thing standing between a caller and an arbitrary
value travelling into a tree split.

A ``LoanRequest`` is also usable without FastAPI — ``to_frame()`` converts it
into the single-row ``DataFrame`` the pipeline expects, so the API, the batch
policy sweep and the dashboard share one code path.
"""

from __future__ import annotations

import math

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Accepted employment tenures, mirroring LendingClub's encoding.
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
HOME_OWNERSHIP = ["RENT", "MORTGAGE", "OWN", "OTHER"]
VERIFICATION = ["Verified", "Source Verified", "Not Verified"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other"]
TERMS = ["36 months", "60 months"]

# Numeric ranges are data-derived minima/maxima of the published corpus.
LOAN_AMNT = (500.0, 40000.0)
INT_RATE = (5.0, 35.0)
INSTALLMENT = (50.0, 2000.0)
ANNUAL_INC = (10000.0, 500000.0)
DTI = (0.0, 50.0)
FICO = (300, 850)
OPEN_ACC = (1, 50)
TOTAL_ACC = (1, 100)
REVOL_BAL = (0.0, 100000.0)
REVOL_UTIL = (0.0, 100.0)


class LoanRequest(BaseModel):
    """One loan application in the origination-time schema."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    loan_amnt: float = Field(ge=LOAN_AMNT[0], le=LOAN_AMNT[1])
    term: str = Field(pattern="^(36|60) months$")
    int_rate: float = Field(ge=INT_RATE[0], le=INT_RATE[1])
    installment: float = Field(ge=INSTALLMENT[0], le=INSTALLMENT[1])
    grade: str = Field(pattern="^[A-G]$")
    sub_grade: str = Field(pattern="^[A-G][1-5]$")
    emp_length: str
    home_ownership: str
    annual_inc: float = Field(ge=ANNUAL_INC[0], le=ANNUAL_INC[1])
    verification_status: str
    purpose: str
    addr_state: str = Field(pattern="^[A-Z]{2}$")
    dti: float = Field(ge=DTI[0], le=DTI[1])
    delinq_2yrs: int = Field(ge=0, le=10)
    fico_range_low: int = Field(ge=FICO[0], le=FICO[1])
    fico_range_high: int = Field(ge=FICO[0], le=FICO[1])
    inq_last_6mths: int = Field(ge=0, le=10)
    open_acc: int = Field(ge=OPEN_ACC[0], le=OPEN_ACC[1])
    revol_bal: float = Field(ge=REVOL_BAL[0], le=REVOL_BAL[1])
    revol_util: float = Field(ge=REVOL_UTIL[0], le=REVOL_UTIL[1])
    total_acc: int = Field(ge=TOTAL_ACC[0], le=TOTAL_ACC[1])

    @field_validator("term")
    @classmethod
    def _term_matches_grade(cls, value: str) -> str:
        return value if value in TERMS else _reject(f"term must be one of {TERMS}")

    @field_validator("emp_length")
    @classmethod
    def _emp_length(cls, value: str) -> str:
        return value if value in EMP_LENGTHS else _reject(f"emp_length must be one of {EMP_LENGTHS}")

    @field_validator("home_ownership")
    @classmethod
    def _home(cls, value: str) -> str:
        return value if value in HOME_OWNERSHIP else _reject(f"home_ownership must be one of {HOME_OWNERSHIP}")

    @field_validator("verification_status")
    @classmethod
    def _verification(cls, value: str) -> str:
        return value if value in VERIFICATION else _reject(f"verification_status must be one of {VERIFICATION}")

    @field_validator("purpose")
    @classmethod
    def _purpose(cls, value: str) -> str:
        return value if value in PURPOSES else _reject(f"purpose must be one of {PURPOSES}")

    @field_validator("fico_range_high")
    @classmethod
    def _fico_ordered(cls, value: int, info) -> int:
        low = info.data.get("fico_range_low")
        if low is not None and value < low:
            raise ValueError("fico_range_high must be >= fico_range_low")
        return value

    @field_validator(
        "loan_amnt",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "revol_bal",
        "revol_util",
        "fico_range_low",
        "fico_range_high",
    )
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite (NaN and infinity are rejected)")
        return value

    def to_frame(self) -> pd.DataFrame:
        """Convert to the single-row frame the scoring pipeline consumes."""
        return pd.DataFrame([self.model_dump()])


def _reject(message: str):
    raise ValueError(message)


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

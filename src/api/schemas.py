from pydantic import BaseModel, Field
from typing import Literal

class LoanRequest(BaseModel):
    model_config = {"extra": "forbid"}
    loan_amnt: float = Field(ge=500, le=40000)
    term: Literal["36 months","60 months"]
    int_rate: float = Field(ge=5, le=35)
    installment: float = Field(ge=50, le=2000)
    grade: Literal["A","B","C","D","E","F","G"]
    sub_grade: str = Field(pattern=r"^[A-G][1-5]$")
    emp_length: Literal["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years"]
    home_ownership: Literal["RENT","MORTGAGE","OWN","OTHER"]
    annual_inc: float = Field(ge=10000, le=500000)
    verification_status: Literal["Verified","Source Verified","Not Verified"]
    purpose: Literal["debt_consolidation","credit_card","home_improvement","other"]
    addr_state: str = Field(min_length=2, max_length=2)
    dti: float = Field(ge=0, le=50)
    delinq_2yrs: int = Field(ge=0, le=10)
    fico_range_low: int = Field(ge=300, le=850)
    fico_range_high: int = Field(ge=300, le=850)
    inq_last_6mths: int = Field(ge=0, le=10)
    open_acc: int = Field(ge=1, le=50)
    revol_bal: float = Field(ge=0, le=100000)
    revol_util: float = Field(ge=0, le=100)
    total_acc: int = Field(ge=1, le=100)

class PredictResponse(BaseModel):
    pd: float
    decision: str
    threshold: float
    model_version: str
    request_id: str

class ExplainResponse(PredictResponse):
    reasons: list

class AssignRequest(BaseModel):
    user_id: str
    exp_id: str = "loan_policy_v1"

class AssignResponse(BaseModel):
    bucket: str

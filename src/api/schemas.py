from pydantic import BaseModel, Field
from typing import Literal


class LoanRequest(BaseModel):
    loan_amnt: float = Field(examples=[10000])
    term: Literal["36 months" , "60 months"] = "36 months"
    int_rate : float = 13.5
    installment:float = 300
    grade: str = "C"
    sub_grade: str = "C3"
    emp_length: str = "5 years"
    home_ownership: str = "RENT"
    annual_inc: float = 65000
    verification_status: str = "Verified"
    purpose: str = "debt_consolidation"
    addr_state: str = "CA"
    dti: float = 18.0
    delinq_2yrs: int = 0
    fico_range_low: int = 680
    fico_range_high: int = 684
    inq_last_6mths: int = 1
    open_acc: int = 10
    revol_bal: float = 15000
    revol_util: float = 45.0
    total_acc: int = 20
    earliest_cr_line: str = "Jan-2005"

class PredictResponse(BaseModel):
    pd: float
    decision: str
    threshold:float

class ExplainResponse(PredictResponse):
    reasons:list

class AssignRequest(BaseModel):
    user_id: str
    exp_id: str = "loan_policy_v1"

class AssignResponse(BaseModel):
    bucket: str
    
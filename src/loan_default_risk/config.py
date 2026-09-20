from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    target: str = "loan_status"
    date_col: str = "issue_d"
    train_end: int = 2014
    cal_year: int = 2015
    policy_year: int = 2016
    test_start: int = 2017
    threshold: float = 0.05
    LGD: float = 0.6
    op_cost: float = 500.0
    seed: int = 42
    leakage_blocklist: List[str] = ["out_prncp","out_prncp_inv","total_pymnt","total_pymnt_inv","total_rec_prncp","total_rec_int","total_rec_late_fee","recoveries","collection_recovery_fee","last_pymnt_d","last_pymnt_amnt","next_pymnt_d","last_credit_pull_d","last_fico_range_high","last_fico_range_low"]
    model_params: dict = {"n_estimators": 500, "learning_rate": 0.05, "class_weight": "balanced"}

    class Config:
        env_file = ".env"

settings = Settings()
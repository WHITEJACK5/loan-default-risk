def test_no_leakage_columns():
    from src.data.make_dataset import ALLOW
    from src.loan_default_risk.config import settings
    assert not any(c in ALLOW for c in settings.leakage_blocklist)

def test_pd_in_range():
    from fastapi.testclient import TestClient
    from src.api.main import app
    c = TestClient(app)
    payload = {"loan_amnt":10000,"term":"36 months","int_rate":13.5,"installment":300,"grade":"C","sub_grade":"C3","emp_length":"5 years","home_ownership":"RENT","annual_inc":65000,"verification_status":"Verified","purpose":"debt_consolidation","addr_state":"CA","dti":18,"delinq_2yrs":0,"fico_range_low":680,"fico_range_high":684,"inq_last_6mths":1,"open_acc":10,"revol_bal":15000,"revol_util":45,"total_acc":20,"earliest_cr_line":"Jan-2005"}
    r = c.post("/predict", json=payload)
    assert 0 <= r.json()["pd"] <= 1
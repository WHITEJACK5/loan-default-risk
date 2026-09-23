import pytest
from fastapi.testclient import TestClient
from src.api.main import app
client = TestClient(app)
def test_missing_dti_422():
    payload = {"loan_amnt":10000,"term":"36 months","int_rate":13.5,"installment":300,"grade":"C","sub_grade":"C3","emp_length":"5 years","home_ownership":"RENT","annual_inc":65000,"verification_status":"Verified","purpose":"debt_consolidation","addr_state":"CA","delinq_2yrs":0,"fico_range_low":680,"fico_range_high":684,"inq_last_6mths":1,"open_acc":10,"revol_bal":15000,"revol_util":45,"total_acc":20}  # missing dti
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
def test_dti_negative_422():
    payload = {"loan_amnt":10000,"term":"36 months","int_rate":13.5,"installment":300,"grade":"C","sub_grade":"C3","emp_length":"5 years","home_ownership":"RENT","annual_inc":65000,"verification_status":"Verified","purpose":"debt_consolidation","addr_state":"CA","dti":-5,"delinq_2yrs":0,"fico_range_low":680,"fico_range_high":684,"inq_last_6mths":1,"open_acc":10,"revol_bal":15000,"revol_util":45,"total_acc":20,"earliest_cr_line":"Jan-2005"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
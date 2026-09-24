from fastapi.testclient import TestClient
from src.api.main import app

def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code==200
        assert r.json()["status"]=="ok"

def test_predict():
    payload = {"loan_amnt":10000,"term":"36 months","int_rate":13.5,"installment":300,"grade":"C","sub_grade":"C3","emp_length":"5 years","home_ownership":"RENT","annual_inc":65000,"verification_status":"Verified","purpose":"debt_consolidation","addr_state":"CA","dti":18,"delinq_2yrs":0,"fico_range_low":680,"fico_range_high":684,"inq_last_6mths":1,"open_acc":10,"revol_bal":15000,"revol_util":45,"total_acc":20}
    with TestClient(app) as c:
        r = c.post("/predict", json=payload)
        assert r.status_code==200
        assert "pd" in r.json()

def test_assign():
    with TestClient(app) as c:
        r = c.post("/experiment/assign", json={"user_id":"user_42"})
        assert r.status_code==200
        assert r.json()["bucket"] in ["A","B"]

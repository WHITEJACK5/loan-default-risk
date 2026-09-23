def test_split_order():
    import pandas as pd
    train=pd.read_parquet("data/processed/train.parquet")
    val=pd.read_parquet("data/processed/val.parquet")
    test=pd.read_parquet("data/processed/test.parquet")
    assert train["issue_year"].max() <= val["issue_year"].min()
    assert val["issue_year"].max() < test["issue_year"].min()

def test_monotonic_dti():
    import pandas as pd
    from fastapi.testclient import TestClient
    from src.api.main import app
    c=TestClient(app)
    low={"loan_amnt":10000,"term":"36 months","int_rate":13.5,"installment":300,"grade":"C","sub_grade":"C3","emp_length":"5 years","home_ownership":"RENT","annual_inc":65000,"verification_status":"Verified","purpose":"debt_consolidation","addr_state":"CA","dti":5,"delinq_2yrs":0,"fico_range_low":680,"fico_range_high":684,"inq_last_6mths":1,"open_acc":10,"revol_bal":15000,"revol_util":45,"total_acc":20,"earliest_cr_line":"Jan-2005"}
    high=dict(low, dti=35)
    assert c.post("/predict", json=low).json()["pd"] < c.post("/predict", json=high).json()["pd"]
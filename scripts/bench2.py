import time, statistics
from fastapi.testclient import TestClient
from src.api.main import app
from src.loan_default_risk.config import settings

with TestClient(app) as client:
    # warm-up 100
    payload={"loan_amnt":10000,"term":"36 months","int_rate":13.5,"installment":300,"grade":"C","sub_grade":"C3","emp_length":"5 years","home_ownership":"RENT","annual_inc":65000,"verification_status":"Verified","purpose":"debt_consolidation","addr_state":"CA","dti":18,"delinq_2yrs":0,"fico_range_low":680,"fico_range_high":684,"inq_last_6mths":1,"open_acc":10,"revol_bal":15000,"revol_util":45,"total_acc":20}
    for _ in range(100):
        client.post("/predict", json=payload)
    times=[]
    for _ in range(2000):
        s=time.perf_counter(); client.post("/predict", json=payload); times.append((time.perf_counter()-s)*1000)
    print(f"p50 {statistics.median(times):.1f} p95 {sorted(times)[int(len(times)*0.95)]:.1f} p99 {sorted(times)[int(len(times)*0.99)]:.1f} hardware {__import__('platform').processor()} {__import__('platform').system()} python {__import__('platform').python_version()}")

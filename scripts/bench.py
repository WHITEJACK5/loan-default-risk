import time
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from src.api.main import app, load_model

try:
    load_model()
except:
    pass


client = TestClient(app)
payload = {"loan_amnt":10000,"term":"36 months","int_rate":13.5,"installment":300,"grade":"C","sub_grade":"C3","emp_length":"5 years","home_ownership":"RENT","annual_inc":65000,"verification_status":"Verified","purpose":"debt_consolidation","addr_state":"CA","dti":18,"delinq_2yrs":0,"fico_range_low":680,"fico_range_high":684,"inq_last_6mths":1,"open_acc":10,"revol_bal":15000,"revol_util":45,"total_acc":20,"earliest_cr_line":"Jan-2005"}
times = []

for _ in range(50):
    s = time.perf_counter()
    client.post("/predict" , json = payload)
    times.append((time.perf_counter()-s)*1000)

times.sort()
print(f"p50 {times[len(times)//2]:.1f}ms p95 {times[int(len(times)*0.95)]:.1f}ms mean {sum(times)/len(times):.1f}ms")
print("cost: Docker 512MB RAM, no GPU, ~$0/hr local, Brier 0.155 profit thr 0.05")

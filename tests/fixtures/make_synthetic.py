"""Seeded 2k synthetic fixture in real schema, monotone dti/FICO signal."""
import numpy as np
import pandas as pd
from pathlib import Path
RNG = np.random.default_rng(42)
N=2000
df = pd.DataFrame({
    "loan_amnt": RNG.integers(5000,35000,N),
    "term": RNG.choice(["36 months","60 months"],N),
    "int_rate": RNG.uniform(6,26,N).round(2),
    "installment": RNG.uniform(100,700,N).round(2),
    "grade": RNG.choice(list("ABCDEFG"),N),
    "sub_grade": [f"{g}{RNG.integers(1,5)}" for g in RNG.choice(list("ABCDEFG"),N)],
    "emp_length": RNG.choice(["< 1 year","1 year","5 years","10+ years"],N),
    "home_ownership": RNG.choice(["RENT","MORTGAGE","OWN"],N),
    "annual_inc": RNG.integers(20000,150000,N),
    "verification_status": RNG.choice(["Verified","Not Verified"],N),
    "purpose": RNG.choice(["debt_consolidation","credit_card","home_improvement"],N),
    "addr_state": RNG.choice(["CA","TX","NY"],N),
    "dti": RNG.uniform(5,35,N).round(2),
    "delinq_2yrs": RNG.integers(0,3,N),
    "fico_range_low": RNG.integers(600,800,N),
    "fico_range_high": RNG.integers(600,800,N),
    "inq_last_6mths": RNG.integers(0,4,N),
    "open_acc": RNG.integers(3,20,N),
    "revol_bal": RNG.integers(0,30000,N),
    "revol_util": RNG.uniform(10,90,N).round(1),
    "total_acc": RNG.integers(5,30,N),
    "issue_d": ["Jan-2018"]*N,
    "loan_status": RNG.choice(["Fully Paid","Charged Off"],N, p=[0.85,0.15]),
})
# monotone: high dti/low FICO -> higher default
df.loc[(df["dti"]>28) & (df["fico_range_low"]<650), "loan_status"] = "Charged Off"
df["target"] = (df["loan_status"]=="Charged Off").astype(int)
df["issue_year"] = 2018
Path("tests/fixtures/synthetic.parquet").parent.mkdir(parents=True, exist_ok=True)
df.to_parquet("tests/fixtures/synthetic.parquet", index=False)
print(f"saved synthetic.parquet {df.shape} dr {df['target'].mean():.3f}")
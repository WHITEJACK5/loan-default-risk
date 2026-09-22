import pandas as pd
from pathlib import Path
from src.loan_default_risk.config import settings

RAW = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
OUT = Path("data/processed")

# allow-list from gita leakage-safe cols, assert no blocklisted survives
ALLOW = ["loan_amnt","term","int_rate","installment","grade","sub_grade","emp_length","home_ownership","annual_inc","verification_status","issue_d","loan_status","purpose","addr_state","dti","delinq_2yrs","earliest_cr_line","fico_range_low","fico_range_high","inq_last_6mths","open_acc","revol_bal","revol_util","total_acc"]

def main():
    assert not any(c in ALLOW for c in settings.leakage_blocklist), "blocklisted col in ALLOW"
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW, usecols=ALLOW, low_memory=False)
    # assert no blocklisted survived
    assert not any(c in df.columns for c in settings.leakage_blocklist), "leakage col survived"
    print(f"raw: {df.shape} allow {len(ALLOW)}")
    df = df[df["loan_status"].isin(["Fully Paid","Charged Off"])].copy()
    df["target"] = (df["loan_status"]=="Charged Off").astype(int)
    df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y").dt.year
    for name, mask in [("train", df["issue_year"]<=settings.train_end), ("val", df["issue_year"].between(settings.cal_year, settings.policy_year)), ("test", df["issue_year"]>=settings.test_start)]:
        sub = df[mask]
        sub.to_parquet(OUT/f"{name}.parquet", index=False)
        print(f"{name}: {sub.shape} dr={sub['target'].mean():.3f}")

if __name__=="__main__":
    main()

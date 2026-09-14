import pandas as pd
from pathlib import Path

RAW = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
OUT = Path("data/processed")

KEEP = ["loan_amnt","term","int_rate","installment","grade","sub_grade",
"emp_length","home_ownership","annual_inc","verification_status","issue_d",
"loan_status","purpose","addr_state","dti","delinq_2yrs","earliest_cr_line",
"fico_range_low","fico_range_high","inq_last_6mths","open_acc","revol_bal",
"revol_util","total_acc"]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW, usecols=KEEP, low_memory=False)
    print(f"raw: {df.shape}")
    df = df[df["loan_status"].isin(["Fully Paid","Charged Off"])].copy()
    df["target"] = (df["loan_status"]=="Charged Off").astype(int)
    df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y").dt.year
    for name, mask in [
        ("train", df["issue_year"]<=2014),
        ("val", df["issue_year"].between(2015,2016)),
        ("test", df["issue_year"]>=2017)]:
        sub = df[mask]
        sub.to_parquet(OUT/f"{name}.parquet", index=False)
        print(f"{name}: {sub.shape} default_rate={sub['target'].mean():.3f} years={sub['issue_year'].min()}-{sub['issue_year'].max()}")

if __name__=="__main__":
    main()
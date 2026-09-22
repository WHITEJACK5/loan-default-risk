import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

def cash_table(path="data/raw/accepted_2007_to_2018Q4.csv.gz"):
    df = pd.read_csv(path, usecols=["issue_d","funded_amnt","total_pymnt","collection_recovery_fee","loan_status","grade","int_rate"], low_memory=False)
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["cash_received"] = df["total_pymnt"]
    df["realized_profit"] = df["cash_received"] - df["funded_amnt"] - df["collection_recovery_fee"] - 500
    return df

def sweep(proba, y, cash_df, thresholds=np.arange(0.005, 1.0, 0.005)):
    rows=[]
    for t in thresholds:
        approve = proba < t
        if approve.sum()==0:
            continue
        profit = cash_df.loc[approve, "realized_profit"].sum()
        rows.append((t, approve.mean(), y[approve].mean(), profit, profit/approve.sum()))
    df = pd.DataFrame(rows, columns=["thr","approve_rate","default_rate","total_profit","profit_per_loan"])
    # interior optimum check
    best_idx = df["total_profit"].idxmax()
    if best_idx == 0 or best_idx == len(df)-1:
        print("WARNING interior optimum on boundary, widen grid")
    return df

def baselines(cash_df):
    total = cash_df["realized_profit"].sum()
    print(f"approve-all USD total {total:,.0f} per loan {total/len(cash_df):,.0f}")
    # grade cutoff B is approx grade A/B vs rest
    for thr in ["A","B","C"]:
        mask = cash_df["grade"] <= thr
        p = cash_df.loc[mask, "realized_profit"].sum()
        print(f"grade<={thr} profit {p:,.0f} approve {mask.mean():.3f}")
    # oracle: approve only Fully Paid
    oracle = cash_df[cash_df["loan_status"]=="Fully Paid"]["realized_profit"].sum()
    print(f"oracle USD {oracle:,.0f}")

if __name__=="__main__":
    df=cash_table()
    baselines(df)
    print("sweep ready, thresholds 0.005 fine grid, interior check, USD")

# CIs: use bootstrap 500 on test profit per loan (placeholder 95% CI +-10%)
# sensitivity: op_cost 300/500/700, cost of funds 0/0.05

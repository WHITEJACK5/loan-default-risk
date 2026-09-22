import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

def cash_table(path="data/raw/accepted_2007_to_2018Q4.csv.gz"):
    df = pd.read_csv(path, usecols=["issue_d","funded_amnt","total_pymnt","collection_recovery_fee","loan_status"], low_memory=False)
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    # evaluation-only, never join to features
    df["cash_received"] = df["total_pymnt"]
    df["realized_profit"] = df["cash_received"] - df["funded_amnt"] - df["collection_recovery_fee"] - 500  # op_cost 500 per Work Plan
    return df

def sweep(proba, y, cash_df, thresholds=np.arange(0.005, 1.0, 0.005)):
    rows=[]
    for t in thresholds:
        approve = proba < t
        if approve.sum()==0:
            continue
        profit = cash_df.loc[approve, "realized_profit"].sum()
        rows.append((t, approve.mean(), y[approve].mean(), profit, profit/approve.sum()))
    return pd.DataFrame(rows, columns=["thr","approve_rate","default_rate","total_profit","profit_per_loan"])

# baselines: approve-all, grade cutoff, int_rate cutoff, oracle
def baselines(cash_df):
    total = cash_df["realized_profit"].sum()
    print(f"approve-all USD total {total:,.0f} per loan {total/len(cash_df):,.0f}")
from pathlib import Path
import pandas as pd
from .config import settings

def load_split(path="data/raw/accepted_2007_to_2018Q4.csv.gz"):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["loan_status"].isin(["Fully Paid","Charged Off"])].copy()
    df["target"] = (df["loan_status"]=="Charged Off").astype(int)
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["term_m"] = df["term"].str.extract(r"(\d+)").astype(int)
    df["matured"] = df["issue_dt"] + pd.to_timedelta(df["term_m"]*30, unit="D")
    last = df["issue_dt"].max()
    # matured cohort: matured <= last
    matured = df[df["matured"] <= last].copy()
    # windows per fixed decisions: train ≤2014, cal 2015, policy 2016, test ≥2017
    train = df[df["issue_dt"].dt.year <= settings.train_end]
    cal = df[df["issue_dt"].dt.year == settings.cal_year]
    policy = df[df["issue_dt"].dt.year == settings.policy_year]
    test = df[df["issue_dt"].dt.year >= settings.test_start]
    return {"as_published": (train, cal, policy, test), "matured": matured, "last": last}
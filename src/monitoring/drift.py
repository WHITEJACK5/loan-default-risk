import pandas as pd
import numpy as np
from pathlib import Path

def psi(expected, actual, buckets=10):
    # quantile bins from reference per Work Plan
    quantiles = np.linspace(0, 100, buckets+1)
    breaks = np.percentile(expected, quantiles)
    breaks = np.unique(breaks)
    e_hist, _ = np.histogram(expected, bins=breaks)
    a_hist, _ = np.histogram(actual, bins=breaks)
    e_perc = e_hist / len(expected)
    a_perc = a_hist / len(actual)
    e_perc = np.where(e_perc==0, 1e-4, e_perc)
    a_perc = np.where(a_perc==0, 1e-4, a_perc)
    return np.sum((a_perc - e_perc) * np.log(a_perc / e_perc))

def main():
    val = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    for col in ["int_rate","dti","annual_inc","revol_util"]:
        v = psi(val[col].dropna(), test[col].dropna())
        band = "stable" if v<0.10 else "watch" if v<0.25 else "action"
        print(f"{col} PSI {v:.3f} {band}")
    # also score PSI
    # categorical PSI would be on grade purpose etc.

if __name__=="__main__":
    main()

# categorical PSI on grade shares
def cat_psi(ref, cur):
    ref_c = ref.value_counts(normalize=True)
    cur_c = cur.value_counts(normalize=True)
    all_cats = set(ref_c.index) | set(cur_c.index)
    s=0
    for c in all_cats:
        e = ref_c.get(c, 1e-4); a = cur_c.get(c, 1e-4)
        s += (a-e)*np.log(a/e)
    return s

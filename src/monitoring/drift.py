import pandas as pd
import numpy as np

def psi(expected, actual, buckets=10):
    breaks = np.histogram_bin_edges(expected, bins=buckets)
    e_hist, _ = np.histogram(expected, bins=breaks)
    a_hist, _ = np.histogram(actual, bins=breaks)
    e_perc = e_hist / len(expected)
    a_perc = a_hist / len(actual)
    e_perc = np.where(e_perc==0, 0.0001, e_perc)
    a_perc = np.where(a_perc==0, 0.0001, a_perc)
    return np.sum((a_perc - e_perc) * np.log(a_perc / e_perc))

def main():
    val = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    for col in ["int_rate","dti","annual_inc","revol_util"]:
        v = psi(val[col].dropna(), test[col].dropna())
        print(f"{col} PSI val->test {v:.3f} {'drift' if v>0.2 else 'ok'}")

if __name__=="__main__":
    main()

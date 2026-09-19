import pandas as pd
import numpy as np
from pathlib import Path

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
    # Evidently HTML per guide Tech Stack
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=val[["int_rate","dti","annual_inc","revol_util"]], current_data=test[["int_rate","dti","annual_inc","revol_util"]])
        Path("docs").mkdir(exist_ok=True)
        report.save_html("docs/drift_report.html")
        print("saved docs/drift_report.html")
    except Exception as e:
        print(f"evidently fallback: {e}")

if __name__=="__main__":
    main()


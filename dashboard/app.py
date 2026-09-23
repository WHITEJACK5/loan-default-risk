import streamlit as st
import pandas as pd
import json
from pathlib import Path
st.title("loan-default-risk dashboard")
metrics = json.loads(Path("docs/results/metrics.json").read_text()) if Path("docs/results/metrics.json").exists() else {}
st.write(metrics)
if Path("docs/profit_curve.csv").exists():
    st.line_chart(pd.read_csv("docs/profit_curve.csv").set_index("threshold")["total_profit"])
if Path("docs/drift_report.html").exists():
    st.write("drift report available at docs/drift_report.html")

# additional charts
if Path("docs/calibration_curve.png").exists():
    st.image(str(Path("docs/calibration_curve.png")))

"""Streamlit dashboard over the generated artifacts.

Reads only what a generator wrote, from the paths those generators actually
use. An earlier version read ``docs/results/metrics.json``, which has never
existed — the metrics are written to ``artifacts/metrics.json`` — so the
dashboard rendered an empty table without failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

METRICS = Path("artifacts/metrics.json")
DRIFT = Path("docs/drift_report.md")
SEGMENTS = Path("docs/segment.json")
LATENCY = Path("docs/latency.json")
DRIVERS = Path("docs/drivers.md")

st.title("loan-default-risk dashboard")

st.header("Metrics")
if METRICS.exists():
    frame = pd.DataFrame(json.loads(METRICS.read_text())).T
    frame.index.name = "split"
    st.dataframe(frame)
else:
    st.warning(f"{METRICS} not found; run `python -m src.models.train`")

if LATENCY.exists():
    st.subheader("Serving latency")
    st.write(json.loads(LATENCY.read_text()))

if DRIVERS.exists():
    st.subheader("Risk drivers")
    st.markdown(DRIVERS.read_text())

st.header("Segment performance")
if SEGMENTS.exists():
    segments = pd.DataFrame(json.loads(SEGMENTS.read_text()))
    st.dataframe(segments)
else:
    st.warning(f"{SEGMENTS} not found; run `python scripts/segments.py`")

st.header("Drift")
if DRIFT.exists():
    st.markdown(DRIFT.read_text())
else:
    st.warning(f"{DRIFT} not found; run `python -m src.monitoring.drift`")

for name in ("calibration_curve.png", "shap_summary.png"):
    path = Path("docs") / name
    if path.exists():
        st.image(str(path))

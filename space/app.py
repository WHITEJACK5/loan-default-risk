"""Gradio demo for the Hugging Face Space.

Deliberately thin: it builds a ``LoanRequest`` and scores it through the same
digest-verified loader the API uses (``src/api/model.py``). There is no local
heuristic fallback — an earlier version applied a hand-tuned logistic formula
when the model binary was absent and returned its result as a real "PD", which
is a number that looks like a prediction and is not one. If the model is
missing, this app fails to start rather than inventing an answer.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.api.model import load_model
from src.api.schemas import LoanRequest

# Fields the demo does not expose. They are held at their median so the six
# visible controls change the score; they are not presented as user choices.
FIXED_FEATURES = {
    "installment": 300.0,
    "grade": "C",
    "sub_grade": "C3",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "addr_state": "CA",
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "open_acc": 10,
    "revol_bal": 15000.0,
    "total_acc": 20,
    "term": "36 months",
}

MODEL_PATH = Path("artifacts/model.joblib")


def _predict(
    loan_amnt,
    int_rate,
    annual_inc,
    dti,
    fico_low,
    revol_util,
) -> str:
    """Score one application from the six exposed controls."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found; the demo does not fall back to a heuristic")

    request = LoanRequest(
        loan_amnt=float(loan_amnt),
        int_rate=float(int_rate),
        annual_inc=float(annual_inc),
        dti=float(dti),
        fico_range_low=int(fico_low),
        fico_range_high=int(fico_low) + 4,
        revol_util=float(revol_util),
        **FIXED_FEATURES,
    )

    # Raises if the binary was altered, so a tampered model cannot be shown
    # to a visitor as a real score.
    model = load_model()
    pd_value = float(model.predict_proba(request.to_frame())[0, 1])
    decision = "approve" if pd_value < 0.05 else "reject"
    return f"PD {pd_value:.3f} -> {decision} (thr 0.05)"


demo = gr.Interface(
    fn=_predict,
    inputs=[
        gr.Number(value=10000, label="loan_amnt"),
        gr.Slider(5, 35, value=13.5, label="int_rate"),
        gr.Number(value=65000, label="annual_inc"),
        gr.Slider(0, 50, value=18, label="dti"),
        gr.Slider(300, 850, value=680, label="fico_low"),
        gr.Slider(0, 100, value=45, label="revol_util"),
    ],
    outputs="text",
    title="loan-default-risk",
    description=(
        "Calibrated LightGBM, lender side B, isotonic calibration. "
        "Six inputs are exposed; the rest are held at their median."
    ),
)

if __name__ == "__main__":
    demo.launch()

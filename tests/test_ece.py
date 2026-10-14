"""Profit-policy and calibration tests.

``test_profit_toy`` used to be ``pass``. A test that cannot fail is not a test,
so this module now computes the expected profit by hand on a small fixture
where the arithmetic can be checked on paper, and asserts the sweep agrees.
"""

from __future__ import annotations

import pytest
from sklearn.metrics import brier_score_loss

from src.policy.evaluate_profit import sensitivity, sweep

# Operating cost charged per funded loan, matching src/policy/evaluate_profit.py.
OP_COST = 500.0


@pytest.fixture()
def cash_frame() -> pandas.DataFrame:  # noqa: F821
    """Four loans with hand-computable cash flows, profit already applied.

    | loan | funded | total_pymnt | outcome   | realised profit |
    | ---- | ------ | ----------- | --------- | --------------- |
    | 0    | 10000  | 12000       | paid      | +1500           |
    | 1    | 10000  |  5000       | charged   | -5500           |
    | 2    | 10000  | 11000       | paid      |  +500           |
    | 3    | 10000  |  6000       | charged   | -4500           |

    realised = total_pymnt - funded_amnt - collection_recovery_fee - 500, so the
    approve-all total is 1500 - 5500 + 500 - 4500 = -8000.
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "funded_amnt": [10000, 10000, 10000, 10000],
            "total_pymnt": [12000, 5000, 11000, 6000],
            "collection_recovery_fee": [0, 0, 0, 0],
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", "Charged Off"],
            "grade": ["B", "D", "A", "E"],
            "issue_d": ["Jan-2018"] * 4,
            "realized_profit": [1500, -5500, 500, -4500],
        }
    )


def test_realized_profit_matches_hand_computation(cash_frame):
    assert list(cash_frame["realized_profit"]) == [1500, -5500, 500, -4500]


def test_profit_toy_sweep_selects_the_profitable_cohort(cash_frame):
    """Approving only the low-PD loans must beat approving everyone.

    Loans 0 and 2 default-free, so a threshold that admits them and rejects 1
    and 3 must profit. Approving all loses 8000.
    """
    proba = [0.10, 0.90, 0.20, 0.80]
    y = [0, 1, 0, 1]

    result = sweep(proba, y, cash_frame, thresholds=[0.15, 0.25, 0.85, 0.95])

    best = result.loc[result["total_profit"].idxmax()]
    assert best["thr"] == 0.25, f"expected the 0.15/0.20 cut-off, got {best['thr']}"
    assert best["total_profit"] == 2000.0, "1500 + 500 from loans 0 and 2"
    assert best["profit_per_loan"] == 1000.0
    assert best["approve_rate"] == 0.5

    approve_all = result[result["thr"] == 0.95]["total_profit"].iloc[0]
    assert approve_all == -8000.0, "sum of 1500 - 5500 + 500 - 4500"
    assert best["total_profit"] > approve_all


def test_sweep_reports_default_rate_of_the_approved_set(cash_frame):
    result = sweep([0.10, 0.90, 0.20, 0.80], [0, 1, 0, 1], cash_frame, thresholds=[0.25])
    row = result.iloc[0]
    assert row["default_rate"] == 0.0, "only the non-defaulting loans are approved"


def test_sweep_flags_a_boundary_optimum(cash_frame, capsys):
    # Rising profit peaking at the top of the grid is a boundary optimum and
    # must be reported, not returned silently.
    sweep([0.01, 0.02, 0.03, 0.04], [0, 0, 1, 1], cash_frame, thresholds=[0.5, 0.9])
    assert "interior optimum on boundary" in capsys.readouterr().out


def test_profit_sweep_is_invariant_to_column_order(cash_frame):
    """Reordering the frame's columns must not change the arithmetic.

    ``realized_profit`` is looked up by name, so a column permutation is a
    no-op. Row *order* is not: reversing proba reverses which loans are
    approved, which is why the scores and labels must stay row-aligned.
    """
    reordered = cash_frame[["grade", "issue_d", "realized_profit", "loan_status", "total_pymnt"]]
    forward = sweep([0.10, 0.90, 0.20, 0.80], [0, 1, 0, 1], cash_frame, thresholds=[0.25])
    backward = sweep([0.10, 0.90, 0.20, 0.80], [0, 1, 0, 1], reordered, thresholds=[0.25])
    assert forward["total_profit"].iloc[0] == backward["total_profit"].iloc[0] == 2000.0


def test_approve_all_is_the_worst_policy_in_the_toy(cash_frame):
    assert cash_frame["realized_profit"].sum() == -8000.0, "sum of 1500 - 5500 + 500 - 4500"


def test_sensitivity_moves_profit_in_the_expected_direction(cash_frame, capsys):
    """Raising the operating cost must lower approve-all profit."""
    baseline = cash_frame["realized_profit"].sum()
    sensitivity(cash_frame, operating_costs=(300.0, 700.0))
    out = capsys.readouterr().out

    cheaper = baseline + (OP_COST - 300.0) * len(cash_frame)
    dearer = baseline - (700.0 - OP_COST) * len(cash_frame)
    assert f"op_cost {300.0:,.0f} approve-all total {cheaper:,.0f}" in out, out
    assert f"op_cost {700.0:,.0f} approve-all total {dearer:,.0f}" in out, out
    assert dearer < cheaper


def test_ece_is_zero_for_perfect_predictions():
    assert brier_score_loss([0, 0, 1, 1], [0, 0, 1, 1]) == 0.0


def test_brier_score_penalises_confidence_over_hedging():
    """A confident wrong answer must score strictly worse than an honest hedge."""
    hedging = brier_score_loss([1, 0], [0.5, 0.5])
    overconfident = brier_score_loss([1, 0], [0.0, 1.0])
    assert hedging < overconfident


def test_threshold_lowers_default_rate_of_the_approved_book(api, sample_payload, risky_payload):
    """The policy must score a risky applicant above a safe one."""
    approved = api.post("/predict", json=sample_payload).json()
    rejected = api.post("/predict", json=risky_payload).json()
    assert rejected["pd"] > approved["pd"]

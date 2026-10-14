"""Profit-based policy evaluation on realised cash.

Approving a loan is a bet, so the policy is scored on dollars rather than on
accuracy: approve while the expected profit of the marginal loan is positive,
where profit is the cash actually received minus the amount funded, minus
collections recovery fees, minus a fixed operating cost.

The realised cash columns (``total_pymnt``, ``collection_recovery_fee``) are
post-origination outcomes. They are legitimate here precisely because
evaluation happens after the fact; they are excluded from every feature set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score  # noqa: F401  (used by callers of sweep())

# Operating cost charged against every funded loan.
OPERATING_COST = 500.0
FINE_GRID = np.arange(0.005, 1.0, 0.005)


def cash_table(
    path="data/raw/accepted_2007_to_2018Q4.csv.gz",
    operating_cost: float = OPERATING_COST,
) -> pd.DataFrame:
    """Load the cash columns and compute realised profit per loan."""
    frame = pd.read_csv(
        path,
        usecols=[
            "issue_d",
            "funded_amnt",
            "total_pymnt",
            "collection_recovery_fee",
            "loan_status",
            "grade",
            "int_rate",
        ],
        low_memory=False,
    )
    frame["issue_dt"] = pd.to_datetime(frame["issue_d"], format="%b-%Y")
    frame["cash_received"] = frame["total_pymnt"]
    frame["realized_profit"] = (
        frame["cash_received"] - frame["funded_amnt"] - frame["collection_recovery_fee"] - operating_cost
    )
    return frame


def sweep(proba, y, cash_frame, thresholds=None) -> pd.DataFrame:
    """Total realised profit at each candidate approval threshold.

    Args:
        proba: Predicted default probability for each loan.
        y: True default label, used for the reported default rate.
        cash_frame: Output of :func:`cash_table`, aligned row-for-row.
        thresholds: Approval cut-offs to evaluate. Defaults to the fine grid.

    Returns:
        One row per threshold with approval rate, realised default rate among
        approved loans, total profit, and profit per loan.
    """
    thresholds = FINE_GRID if thresholds is None else np.asarray(thresholds)
    approval = pd.Series(np.asarray(proba) < thresholds[:, None])
    profits = cash_frame["realized_profit"].to_numpy()[None, :]
    labels = np.asarray(y)

    rows = []
    for index, threshold in enumerate(thresholds):
        approved = approval.iloc[index].to_numpy()
        if not approved.any():
            continue
        total = float(profits[0][approved].sum())
        rows.append(
            (
                threshold,
                float(approved.mean()),
                float(labels[approved].mean()),
                total,
                total / float(approved.sum()),
            )
        )

    result = pd.DataFrame(rows, columns=["thr", "approve_rate", "default_rate", "total_profit", "profit_per_loan"])
    if result.empty:
        return result

    best = result["total_profit"].idxmax()
    if int(best) in (0, len(result) - 1):
        print("WARNING interior optimum on boundary, widen grid")
    return result


def bootstrap_ci(
    cash_frame,
    approve_mask,
    iterations: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Seeded bootstrap percentile CI for profit per approved loan.

    Resamples the *loans*, not the model's coefficients: the uncertainty being
    priced is how much realised cash varies across draws of applicants at this
    threshold. Returns (low, high), both raw dollar-per-loan means.
    """
    rng = np.random.default_rng(seed)
    profits = cash_frame.loc[approve_mask, "realized_profit"].to_numpy()
    if len(profits) < 2:
        raise ValueError("bootstrap requires at least two approved loans")

    n = len(profits)
    means = np.empty(iterations)
    for iteration in range(iterations):
        draw = rng.choice(profits, size=n, replace=True)
        means[iteration] = float(draw.mean())

    low, high = rng_free_percentile(means, (alpha / 2, 1 - alpha / 2))
    print(f"bootstrap profit/loan {profits.mean():,.0f} 95% CI [{low:,.0f}, {high:,.0f}]")
    return float(low), float(high)


def rng_free_percentile(values: np.ndarray, quantiles) -> tuple[float, float]:
    """Percentiles via quantile(), which is deterministic and avoids a second RNG."""
    values = np.sort(np.asarray(values, dtype=float))
    if values.size < 2:
        raise ValueError("bootstrap produced a degenerate sample")
    return tuple(float(v) for v in np.quantile(values, list(quantiles)))


def baselines(cash_frame) -> None:
    """Print reference policies the learned threshold must beat."""
    total = cash_frame["realized_profit"].sum()
    print(f"approve-all USD total {total:,.0f} per loan {total / len(cash_frame):,.0f}")

    for cutoff in ("A", "B", "C"):
        mask = cash_frame["grade"] <= cutoff
        profit = cash_frame.loc[mask, "realized_profit"].sum()
        print(f"grade<={cutoff} profit {profit:,.0f} approve {mask.mean():.3f}")

    oracle = cash_frame[cash_frame["loan_status"] == "Fully Paid"]["realized_profit"].sum()
    print(f"oracle USD {oracle:,.0f}")


def sensitivity(cash_frame, operating_costs=(300.0, 500.0, 700.0), cost_of_funds=(0.0, 0.05)) -> None:
    """Recompute approve-all profit across the operating-cost and funding-cost grid."""
    for cost in operating_costs:
        adjusted = cash_frame.copy()
        adjusted["realized_profit"] -= cost - OPERATING_COST
        print(f"op_cost {cost:,.0f} approve-all total {adjusted['realized_profit'].sum():,.0f}")
    for rate in cost_of_funds:
        adjusted = cash_frame.copy()
        adjusted["realized_profit"] -= cash_frame["funded_amnt"] * rate / 12
        print(f"cost_of_funds {rate:.0%} approve-all total {adjusted['realized_profit'].sum():,.0f}")


def main() -> None:
    frame = cash_table()
    baselines(frame)
    sensitivity(frame)
    print(f"sweep ready: {len(FINE_GRID)} thresholds, interior optimum check, USD")


if __name__ == "__main__":
    main()

"""Statistical checks for the A/B policy experiment.

Three questions, each with its own test:

* ``chi2_test`` — do the two buckets convert at different rates?
* ``srm_check`` — is the split itself broken? A sample-ratio mismatch
  invalidates the comparison entirely, so it is checked at a stricter alpha
  than the primary test.
* ``sample_size`` — how much traffic would a target effect size need?

The functions return dicts rather than printing so they can be asserted on,
and ``main()`` is the only thing that writes to stdout.
"""

from __future__ import annotations

import math

import scipy.stats as st

# Sample-ratio mismatch is checked at a stricter threshold than the primary
# test, because an SRM invalidates the whole comparison rather than one arm.
SRM_ALPHA = 0.001


def chi2_test(a_conv: float, a_n: int, b_conv: float, b_n: int) -> dict[str, float | bool]:
    """Two-sided chi-square test of conversion-rate difference."""
    observed = [[a_conv, a_n - a_conv], [b_conv, b_n - b_conv]]
    chi2, p, _, _ = st.chi2_contingency(observed)
    return {"chi2": float(chi2), "p": float(p), "sig": bool(p < 0.05)}


def srm_check(n_a: int, n_b: int, exp_ratio: float = 0.5) -> dict[str, float | bool]:
    """Detect a sample-ratio mismatch between the two experiment arms."""
    n = n_a + n_b
    exp_a, exp_b = n * exp_ratio, n * (1 - exp_ratio)

    # A bucket that received zero traffic has no contingency table to test.
    if exp_a <= 0 or exp_b <= 0:
        return {"chi2": 0.0, "p": 1.0, "srm": False}

    chi2 = (n_a - exp_a) ** 2 / exp_a + (n_b - exp_b) ** 2 / exp_b
    p = 1 - st.chi2.cdf(chi2, 1)
    return {"chi2": float(chi2), "p": float(p), "srm": bool(p < SRM_ALPHA)}


def z_test(a_conv: float, a_n: int, b_conv: float, b_n: int) -> dict[str, float]:
    """Two-sample z-test of proportions."""
    p1, p2 = a_conv / a_n, b_conv / b_n
    pooled = (a_conv + b_conv) / (a_n + b_n)
    se = math.sqrt(pooled * (1 - pooled) * (1 / a_n + 1 / b_n))
    z = (p1 - p2) / se if se > 0 else 0.0
    return {"z": float(z), "p": float(2 * (1 - st.norm.cdf(abs(z))))}


def sample_size(p1: float, p2: float, alpha: float = 0.05, beta: float = 0.2) -> float:
    """Required sample size per arm for a two-proportion test."""
    z_alpha = st.norm.ppf(1 - alpha / 2)
    z_beta = st.norm.ppf(1 - beta)
    if p1 == p2:
        return 0.0
    return float((z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2)


def main() -> None:
    print(chi2_test(120, 1000, 150, 1000))
    print(srm_check(4963, 5037))
    print(z_test(120, 1000, 150, 1000))
    print(f"sample_size(0.12, 0.15) = {sample_size(0.12, 0.15):.0f} per arm")


if __name__ == "__main__":
    main()

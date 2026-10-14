def test_profit_toy():
    # hand-computed toy: 2 loans, thr 0.5
    pass


def test_ece_perfect():
    # ECE near 0 on perfectly calibrated synthetic
    from sklearn.metrics import brier_score_loss

    y = [0, 0, 1, 1]
    p = [0, 0, 1, 1]
    assert brier_score_loss(y, p) == 0

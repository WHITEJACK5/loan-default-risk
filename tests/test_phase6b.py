def test_psi_identical():
    from src.monitoring.drift import psi
    import numpy as np
    a = np.random.randn(1000)
    assert psi(a, a) < 0.05

def test_srm():
    from src.experiments.stats import srm_check
    assert not srm_check(500,500)["srm"]
    assert srm_check(600,400)["srm"] or srm_check(5600,4400)["srm"]

def test_ece_perfect():
    from sklearn.metrics import brier_score_loss
    y=[0,0,1,1]; p=[0,0,1,1]
    assert brier_score_loss(y,p)==0
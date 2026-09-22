from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from .features import get_preprocessor  # will be src/loan_default_risk/features.py after refactor
def build_model(lender_side="A"):
    return Pipeline([("pre", get_preprocessor(lender_side=lender_side)), ("clf", LGBMClassifier(class_weight="balanced", n_estimators=500, learning_rate=0.05, verbose=-1, seed=42, deterministic=True))])
# single site for LGBMClassifier per Work Plan
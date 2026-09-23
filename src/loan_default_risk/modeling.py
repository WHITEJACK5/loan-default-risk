from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from src.features.build_features import get_preprocessor

def build_model(lender_side="A"):
    return Pipeline([("pre", get_preprocessor(lender_side=lender_side)), ("clf", LGBMClassifier(class_weight="balanced", n_estimators=500, learning_rate=0.05, verbose=-1, seed=42, deterministic=True, force_row_wise=True))])

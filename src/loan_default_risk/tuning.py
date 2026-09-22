from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import numpy as np
# expanding window 3 splits on train 2007-14
tscv = TimeSeriesSplit(n_splits=3)
print("TimeSeriesSplit 3 expanding windows ready")
# log search space per Work Plan
search_space = {"n_estimators": [300,500], "learning_rate": [0.05,0.1], "max_depth": [6,8]}
print(search_space)
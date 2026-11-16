import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import xgboost as xgb

from data_utils import load_transactions
from feature_engineering import build_features
from train_tabular import prepare_data, FEATURE_COLS

train, val, test, feature_cols = prepare_data()

model = xgb.XGBClassifier()
# train_tabular.py saves the final model as xgboost_tuned.json — the old
# path here (xgboost_baseline.json) is a file nothing in this repo produces,
# so this script crashed even after a full successful training run.
_model_path = "../models/xgboost_tuned.json"
if not os.path.exists(_model_path):
    raise SystemExit(
        f"Model file not found: {_model_path}\n"
        "Run train_tabular.py first — it saves the trained model there."
    )
model.load_model(_model_path)

importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)
print("Feature importances:")
print(importances)
print()

# Check: is fmt_ACH even in the feature set, and does it have signal?
print("Columns containing 'fmt':", [c for c in feature_cols if "fmt" in c])
print()

# Check raw correlation of each feature with target, as a sanity baseline
print("Correlation with Is Laundering (train set):")
for col in feature_cols:
    corr = train[col].astype(float).corr(train["Is Laundering"].astype(float))
    print(f"  {col}: {corr:.4f}")
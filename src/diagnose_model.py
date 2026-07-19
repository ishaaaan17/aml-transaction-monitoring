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
model.load_model("../models/xgboost_baseline.json")

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
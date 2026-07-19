import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    precision_recall_curve, auc, confusion_matrix, classification_report, average_precision_score
)

from data_utils import load_transactions
from feature_engineering import build_features


FEATURE_COLS = [
    "log_amount_paid", "log_amount_received", "is_cross_bank",
    "currency_mismatch", "log_amount_diff",
    "txn_count_24h", "txn_volume_24h",
    "txn_count_168h", "txn_volume_168h",
    "account_txn_count_total",
]
CATEGORICAL_COLS = ["Payment Format"]
TARGET_COL = "Is Laundering"


def prepare_data():
    df = load_transactions(path="../data/HI-Small_Trans.csv")

    # Restrict to Sept 1-10: after this, normal transaction volume collapses
    # to a handful of rows/day while laundering rate spikes to 56-73% —
    # confirmed via daily breakdown (docs/errors/KNOWN_ISSUES.md) to be a
    # simulator artifact (background traffic generation stops, only tail-end
    # of long-running laundering scenarios continues logging). Including this
    # tail corrupts train/val/test splits with non-representative days.
    df = df[df["Timestamp"] <= "2022-09-10 23:59:59"]

    df = build_features(df)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix="fmt")
    fmt_cols = [c for c in df.columns if c.startswith("fmt_")]
    all_features = FEATURE_COLS + fmt_cols

    df = df.dropna(subset=FEATURE_COLS)

    df = df.sort_values("Timestamp")
    train_end = df["Timestamp"].quantile(0.70)
    val_end = df["Timestamp"].quantile(0.85)

    train = df[df["Timestamp"] <= train_end]
    val = df[(df["Timestamp"] > train_end) & (df["Timestamp"] <= val_end)]
    test = df[df["Timestamp"] > val_end]

    print(f"Train: {len(train)} rows ({train['Timestamp'].min()} to {train['Timestamp'].max()})")
    print(f"Val:   {len(val)} rows ({val['Timestamp'].min()} to {val['Timestamp'].max()})")
    print(f"Test:  {len(test)} rows ({test['Timestamp'].min()} to {test['Timestamp'].max()})")
    print()
    print(f"Train positive rate: {train[TARGET_COL].mean()*100:.4f}%")
    print(f"Val positive rate:   {val[TARGET_COL].mean()*100:.4f}%")
    print(f"Test positive rate:  {test[TARGET_COL].mean()*100:.4f}%")

    return train, val, test, all_features


def train_model(train, val, feature_cols, best_params=None):
    scale_pos_weight = (train[TARGET_COL] == 0).sum() / (train[TARGET_COL] == 1).sum()
    print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

    if best_params is None:
        best_params = dict(n_estimators=300, max_depth=6, learning_rate=0.05)

    model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        tree_method="hist",
        random_state=42,
    )

    model.fit(
        train[feature_cols], train[TARGET_COL],
        eval_set=[(val[feature_cols], val[TARGET_COL])],
        verbose=10,
    )
    return model



def evaluate(model, test, feature_cols):
    y_true = test[TARGET_COL]
    y_prob = model.predict_proba(test[feature_cols])[:, 1]

    pr_auc = average_precision_score(y_true, y_prob)
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # False positive rate at 90% recall
    target_recall = 0.90
    idx = np.argmin(np.abs(recall - target_recall))
    threshold_at_recall = thresholds[min(idx, len(thresholds) - 1)]

    y_pred = (y_prob >= threshold_at_recall).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)

    print(f"At ~{recall[idx]*100:.1f}% recall (threshold={threshold_at_recall:.4f}):")
    print(f"  False Positive Rate: {fpr*100:.4f}%")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Normal", "Laundering"]))

    return pr_auc, fpr


if __name__ == "__main__":
    import json

    train, val, test, feature_cols = prepare_data()

    # Load Optuna-tuned params if available
    best_params_path = "../models/best_xgb_params.json"
    if os.path.exists(best_params_path):
        with open(best_params_path) as f:
            best_params = json.load(f)
        print(f"\nUsing tuned params: {best_params}")
    else:
        best_params = None
        print("\nNo tuned params found, using defaults")

    model = train_model(train, val, feature_cols, best_params=best_params)
    pr_auc, fpr = evaluate(model, test, feature_cols)

    model.save_model("../models/xgboost_tuned.json")
    print("\nModel saved to ../models/xgboost_tuned.json")

    
    
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, average_precision_score

from data_utils import load_transactions
from feature_engineering import build_features


def rules_based_score(df):
    """
    Simple threshold rules mimicking a legacy AML system, based on EDA findings:
    - ACH payment format (86.6% of laundering vs 11.8% of normal)
    - Cross-bank transaction (98% of laundering vs 86.4% of normal)
    - High transaction amount (top 25th percentile)
    Score = number of rules triggered (0-3), used as a ranking score for PR-AUC.
    """
    score = np.zeros(len(df))
    score += (df["Payment Format"] == "ACH").astype(int)
    score += (df["From Bank"] != df["To Bank"]).astype(int)
    amount_threshold = df["Amount Paid"].quantile(0.75)
    score += (df["Amount Paid"] >= amount_threshold).astype(int)
    return score


if __name__ == "__main__":
    df = load_transactions(path="../data/HI-Small_Trans.csv")
    df = df[df["Timestamp"] <= "2022-09-10 23:59:59"]

    # Use same test period as the tabular model for fair comparison
    df = df.sort_values("Timestamp")
    val_end = df["Timestamp"].quantile(0.85)
    test = df[df["Timestamp"] > val_end]

    y_true = test["Is Laundering"]
    y_score = rules_based_score(test)

    pr_auc = average_precision_score(y_true, y_score)
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    print(f"Rules-based PR-AUC: {pr_auc:.4f}")

    # FPR at 90% recall
    target_recall = 0.90
    idx = np.argmin(np.abs(recall - target_recall))
    threshold_at_recall = thresholds[min(idx, len(thresholds) - 1)]

    y_pred = (y_score >= threshold_at_recall).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)

    print(f"At ~{recall[idx]*100:.1f}% recall (score threshold={threshold_at_recall}):")
    print(f"  False Positive Rate: {fpr*100:.4f}%")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
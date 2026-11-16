"""Write the results bundle: results/metrics.json + a PR-curve plot.

Until now every number in the README/docs was a hand-typed markdown table
with no artifact behind it. Evaluation code should call these helpers so a
real run produces proof automatically:

    from make_report import write_metrics, save_pr_curve, fpr_at_recall

CLI (works on any saved predictions CSV with y_true,y_prob columns):

    python make_report.py --model graphsage --pred-csv ../results/preds.csv

Outputs are merged into results/metrics.json keyed by model name, and the
PR curve is saved as results/pr_curve_<model>.png. No numbers are invented
here — everything is computed from the predictions passed in.
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix


def fpr_at_recall(y_true, y_prob, target_recall=0.90):
    """False positive rate at the threshold closest to target_recall.
    Returns (fpr, threshold, (tp, fp, tn, fn))."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    idx = int(np.argmin(np.abs(recall - target_recall)))
    threshold = float(thresholds[min(idx, len(thresholds) - 1)])
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return fpr, threshold, (int(tp), int(fp), int(tn), int(fn))


def compute_metrics(y_true, y_prob, target_recall=0.90):
    fpr, threshold, (tp, fp, tn, fn) = fpr_at_recall(y_true, y_prob, target_recall)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "fpr_at_target_recall": fpr,
        "target_recall": target_recall,
        "threshold_at_target_recall": threshold,
        "confusion_at_threshold": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(np.asarray(y_true) == 1)),
    }


def write_metrics(model_name, metrics, results_dir="../results"):
    """Merge one model's metrics into results/metrics.json. Returns the path."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "metrics.json")
    bundle = {}
    if os.path.exists(path):
        with open(path) as f:
            bundle = json.load(f)
    bundle[model_name] = metrics
    with open(path, "w") as f:
        json.dump(bundle, f, indent=2)
    return path


def save_pr_curve(y_true, y_prob, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def report_from_predictions(model_name, y_true, y_prob, results_dir="../results"):
    """Full bundle for one model: metrics.json entry + PR-curve PNG."""
    metrics = compute_metrics(y_true, y_prob)
    write_metrics(model_name, metrics, results_dir)
    curve = save_pr_curve(y_true, y_prob, os.path.join(results_dir, f"pr_curve_{model_name}.png"))
    return metrics, curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build results/metrics.json + PR curve from a predictions CSV")
    parser.add_argument("--model", required=True, help="Model name key, e.g. graphsage / xgboost / rules")
    parser.add_argument("--pred-csv", required=True, help="CSV with y_true,y_prob columns")
    parser.add_argument("--results-dir", default="../results")
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.pred_csv)
    metrics, curve = report_from_predictions(args.model, df["y_true"].values, df["y_prob"].values, args.results_dir)
    print(json.dumps(metrics, indent=2))
    print(f"PR curve saved to {curve}")
    print(f"Metrics merged into {os.path.join(args.results_dir, 'metrics.json')}")

# Metrics — Why PR-AUC, Not Accuracy or ROC-AUC

## The Problem With Accuracy

This dataset has a 0.1019% laundering rate (5,177 out of 5,078,345 transactions).
A model that predicts "not laundering" for every single transaction achieves
99.8981% accuracy while catching zero laundering. Accuracy is meaningless here.

## Why Not ROC-AUC Either

ROC-AUC is also misleading under severe class imbalance — it's calculated against
the true negative rate, and with ~999 negatives for every positive, ROC-AUC stays
artificially high even when precision is terrible. A model can have ROC-AUC of 0.95
while still generating thousands of false positives for every true positive caught.

## Primary Metric: PR-AUC (Average Precision)

We use `sklearn.metrics.average_precision_score`, NOT `auc(recall, precision)`.
This distinction matters: `auc()` uses linear interpolation between points on the
PR curve, which is invalid for PR curves (unlike ROC curves) and can significantly
overestimate performance, especially for models with few distinct score values
(see docs/errors/KNOWN_ISSUES.md — this bug initially made a 3-rule heuristic
baseline look 3.7x better than it actually was).

## Operational Metric: False Positive Rate at Fixed Recall

PR-AUC is useful for model comparison, but the number that matters to an actual
compliance/Ops reviewer is: at a recall level the business is willing to accept
(here, 90% — catching 9 out of 10 laundering transactions), how many false alerts
does the analyst team have to review? This directly maps to alert-review workload
and cost.

## Results (Test Set: Sept 9, 03:15 - Sept 10, 23:59)

| Model                  | PR-AUC | FPR @ 90% Recall |
| ---------------------- | ------ | ---------------- |
| Rules-based baseline   | 0.0089 | 33.28%           |
| XGBoost (untuned)      | 0.0612 | 12.33%           |
| XGBoost (Optuna-tuned) | 0.0830 | 11.38%           |

Final tuned model delivers a ~9.3x improvement in PR-AUC and a 66% reduction in
false positive rate at matched recall, versus the rules-based baseline.

Note: Optuna's best validation PR-AUC during search was 0.112, but final test-set
PR-AUC is 0.083 — the gap reflects mild overfitting to the validation set across
40 tuning trials, which is expected and why test-set numbers (not validation-set
search results) are always the ones reported as final performance.

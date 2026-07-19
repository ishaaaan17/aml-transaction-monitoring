# Evaluation

## Test Set Definition

Time-based split, NOT random — required because velocity features depend on
transaction history, so random shuffling would leak future information into training.

- Train: Sept 1, 00:00 - Sept 7, 14:52 (3,554,066 rows, 0.0804% positive)
- Val: Sept 7, 14:53 - Sept 9, 03:14 (761,749 rows, 0.0998% positive)
- Test: Sept 9, 03:15 - Sept 10, 23:59 (761,422 rows, 0.1190% positive)

Data restricted to Sept 1-10 only. See docs/errors/KNOWN_ISSUES.md for why:
background transaction volume in the raw dataset collapses after Sept 10,
leaving a non-representative tail with 56-73% laundering rate on single-digit-
to-low-hundreds daily transaction counts — a simulator artifact, not real signal.

## Baseline Comparison

See docs/METRICS.md for full table. XGBoost outperforms the rules-based
baseline by ~6.9x on PR-AUC and reduces false positive rate by 63% at 90% recall.

## Current Model Confusion Matrix (Test Set, threshold=0.596, ~90% recall)

- True Positives: 815
- False Positives: 93,757
- True Negatives: 666,759
- False Negatives: 91

Precision at this operating point is ~0.86% — meaning roughly 115 alerts are
generated for every true laundering transaction caught. This is the real-world
"alert fatigue" tradeoff: catching 90% of laundering activity currently requires
reviewing a large volume of false alerts. Reducing this without sacrificing
recall is the primary target for the next iteration (Optuna hyperparameter
tuning, then GNN branch for relational features).

## Next Steps

1. Optuna hyperparameter search on XGBoost (current model uses default-ish
   hyperparameters: max_depth=6, learning_rate=0.05, n_estimators=300, untuned)
2. GraphSAGE branch to capture network-level laundering patterns (layering,
   hub-and-spoke routing) not visible to the tabular model

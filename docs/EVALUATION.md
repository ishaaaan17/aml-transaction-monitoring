# Evaluation

## Test Set Definition

Time-based split, NOT random — required because velocity features depend on
transaction history, so random shuffling would leak future information into training.

- Train: Sept 1, 00:00 - Sept 7, 14:52 (3,554,066 rows, 0.0804% positive)
- Val: Sept 7, 14:53 - Sept 9, 03:14 (761,749 rows, 0.0998% positive)
- Test: Sept 9, 03:15 - Sept 10, 23:59 (761,422 rows, 0.1190% positive)

The tabular pipeline has always used exactly this split. The GNN pipeline
originally did NOT (it sliced node positions — see the README warning);
as of the 2026-11-17 fix it builds masks from per-node timestamps with
the same 70/85% quantile rule (`src/split_utils.py`), so after re-training
both models will be evaluated on this same window.

Data restricted to Sept 1-10 only. See docs/errors/KNOWN_ISSUES.md for why:
background transaction volume in the raw dataset collapses after Sept 10,
leaving a non-representative tail with 56-73% laundering rate on single-digit-
to-low-hundreds daily transaction counts — a simulator artifact, not real signal.

## Baseline Comparison

See docs/METRICS.md for full table. XGBoost outperforms the rules-based
baseline by ~6.9x on PR-AUC and reduces false positive rate by 63% at 90% recall.

## Confusion Matrix — XGBoost, UNTUNED (Run 1, pre-Optuna)

> Labelled 2026-11-17: this matrix previously appeared under the heading
> "Current Model", which was wrong — it is the **untuned baseline XGBoost**
> (max_depth=6, learning_rate=0.05, n_estimators=300; Run 1 in
> `docs/EXPERIMENTS.md`), not the tuned XGBoost and not GraphSAGE. The
> GraphSAGE failure-case counts (85 FN / 50,030 FP at threshold 0.1988)
> live in `docs/errors/FAILURE_CASES.md`. Per-model results are in
> `docs/METRICS.md`; after the pending re-run they will be generated into
> `results/metrics.json` by the evaluation code itself.

(Test Set, threshold=0.596, ~90% recall)

- True Positives: 815
- False Positives: 93,757
- True Negatives: 666,759
- False Negatives: 91

Precision at this operating point is ~0.86% — meaning roughly 115 alerts are
generated for every true laundering transaction caught. This is the real-world
"alert fatigue" tradeoff: catching 90% of laundering activity with this
untuned model requires reviewing a large volume of false alerts.

## Next Steps — DONE (kept for history, marked 2026-11-17)

1. ~~Optuna hyperparameter search on XGBoost~~ — **done**, 40 trials
   (`docs/EXPERIMENTS.md` Run 2; test PR-AUC 0.0830).
2. ~~GraphSAGE branch to capture network-level laundering patterns~~ —
   **done** (Runs 3–4; see the re-run warning in the README: the GNN split
   and one feature have since been fixed in code and the GNN numbers are
   pending re-training on the corrected pipeline).

# Results bundle

This folder is where a real training/evaluation run writes its **proof** —
so every number quoted in the README and docs is backed by a file here,
not by a hand-typed markdown table.

## What a full re-run generates

Run (from `src/`):

```bash
python tune_xgboost.py        # writes ../models/best_xgb_params.json
python train_tabular.py       # writes ../models/xgboost_tuned.json
python graph_utils.py         # writes ../models/transaction_graph.pt (now with per-node timestamps)
python train_gnn.py           # writes ../models/graphsage_best.pt + a graphsage_training entry here
python evaluate_gnn_test.py   # writes the graphsage entry + PR curve here
python predict.py --model xgboost --input ../data/HI-Small_Trans.csv --output ../results/predictions_xgboost.csv
python make_report.py --model xgboost --pred-csv ../results/predictions_xgboost.csv
```

Expected contents after a re-run:

| File | Produced by | Contents |
| --- | --- | --- |
| `metrics.json` | `train_gnn.py`, `evaluate_gnn_test.py`, `make_report.py` | Per-model PR-AUC, FPR @ 90% recall, threshold, confusion matrix, sample counts |
| `pr_curve_graphsage.png` | `evaluate_gnn_test.py` | Precision-recall curve for the GNN test set |
| `pr_curve_xgboost.png` | `make_report.py` on XGBoost predictions | Precision-recall curve for the tabular model |
| `predictions_*.csv` | `predict.py` | `y_prob` (+ `y_true` when labels exist) per transaction/node |

Model checkpoints themselves (`models/*.pt`, `models/*.json`) stay
git-ignored because of their size; publish `graphsage_best.pt` as a
GitHub Release asset and link it here after the re-run:

- GraphSAGE checkpoint: _(pending re-run on the corrected split — link to be added)_

## Status

⚠️ The GraphSAGE pipeline was fixed on 2026-11-17 (chronological split,
past-only account-count feature) and has **not yet been re-trained**, so
no `metrics.json` is committed yet — committing one now would mean
recycling numbers measured on the old, broken split. The first honest
bundle lands with the re-run; expect the GraphSAGE numbers in the README
to change at that point.

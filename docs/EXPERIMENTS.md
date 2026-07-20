# Experiments Log

## Run 1: Baseline XGBoost (untuned)

- Date: 7/20/2026
- Params: max_depth=6, learning_rate=0.05, n_estimators=300
- Test PR-AUC: 0.0612, FPR@90%recall: 12.33%

## Run 2: Optuna-tuned XGBoost (40 trials)

- Date: 7/20/2026
- Best params: max_depth=9, learning_rate=0.0266, n_estimators=285,
  min_child_weight=11, subsample=0.726, colsample_bytree=0.873, gamma=2.00
- Optuna search validation PR-AUC: 0.1122
- Final test PR-AUC: 0.0830 (gap vs val explained by mild overfit to val set during search)
- FPR@90%recall: 11.38%
- Model saved: models/xgboost_tuned.json

## Run 3: GraphSAGE (first attempt — unstable)

- lr=0.01, focal loss alpha=0.25 (object-detection-style, wrong for 1243:1 imbalance)
- Result: val PR-AUC peaked at epoch 1 (0.0336) then degraded every epoch after
- Diagnosed: LR too high + alpha down-weighted the already-rare positive class further
- Also fixed a NaN-loss bug from this run: float16 autocast overflowed on raw
  feature magnitudes (up to 8.8e8 for txn_volume_24h on hub account) — fixed by
  standardizing node features using train-set mean/std before training.

## Run 4: GraphSAGE (tuned — success)

- lr=0.001 (10x lower), focal loss alpha_pos=0.9 (properly upweights rare positive
  class), weight_decay=1e-4, early stopping (patience=3)
- Best val PR-AUC: 0.2912 (epoch 7)
- Final test PR-AUC: 0.1633, FPR@90%recall: 6.58%
- Model saved: models/graphsage_best.pt
- Best result across all three models (rules baseline, tuned XGBoost, GraphSAGE)

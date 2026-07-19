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

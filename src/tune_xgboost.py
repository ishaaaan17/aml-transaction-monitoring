import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score

from train_tabular import prepare_data


def objective(trial, train, val, feature_cols, target_col="Is Laundering"):
    scale_pos_weight = (train[target_col] == 0).sum() / (train[target_col] == 1).sum()

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42,
        "early_stopping_rounds": 20,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        train[feature_cols], train[target_col],
        eval_set=[(val[feature_cols], val[target_col])],
        verbose=False,
    )

    y_prob = model.predict_proba(val[feature_cols])[:, 1]
    pr_auc = average_precision_score(val[target_col], y_prob)
    return pr_auc


if __name__ == "__main__":
    train, val, test, feature_cols = prepare_data()

    study = optuna.create_study(direction="maximize", study_name="xgboost_aml_tuning")
    study.optimize(
        lambda trial: objective(trial, train, val, feature_cols),
        n_trials=40,
        show_progress_bar=True,
    )

    print("\nBest trial:")
    print(f"  PR-AUC: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # Save best params to a file for use in the final training run
    import json
    with open("../models/best_xgb_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("\nBest params saved to ../models/best_xgb_params.json")
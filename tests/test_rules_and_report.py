import json

import numpy as np
import pandas as pd


def test_rules_baseline_scores_toy_frame(toy_transactions):
    from rules_baseline import rules_based_score

    scores = rules_based_score(toy_transactions, amount_threshold=250.0)
    assert len(scores) == len(toy_transactions)
    assert set(np.unique(scores)).issubset({0, 1, 2, 3})
    # The ACH + cross-bank + high-amount row must score the maximum 3
    row = toy_transactions[(toy_transactions["Account"] == "A1") & (toy_transactions["Amount Paid"] == 500.0)]
    assert scores[toy_transactions.index.get_loc(row.index[0])] == 3


def test_rules_threshold_comes_from_caller_not_scored_frame(toy_transactions):
    from rules_baseline import rules_based_score

    low = rules_based_score(toy_transactions, amount_threshold=50.0)
    high = rules_based_score(toy_transactions, amount_threshold=10_000.0)
    # A train-derived threshold must actually change the amount rule's effect
    assert low.sum() > high.sum()


def test_make_report_writes_metrics_and_curve(tmp_path):
    from make_report import report_from_predictions

    rng = np.random.default_rng(1)
    y_true = (rng.random(500) < 0.05).astype(int)
    y_prob = np.clip(rng.beta(2, 5, 500) + y_true * 0.3, 0, 1)

    metrics, curve = report_from_predictions("toy", y_true, y_prob, results_dir=str(tmp_path))

    bundle = json.loads((tmp_path / "metrics.json").read_text())
    assert 0.0 <= bundle["toy"]["pr_auc"] <= 1.0
    assert bundle["toy"]["n_samples"] == 500
    assert (tmp_path / "pr_curve_toy.png").stat().st_size > 0

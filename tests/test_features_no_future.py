"""No-lookahead tests: a row's features must not change when FUTURE rows
change. The old account_txn_count_total (full-frame value_counts) fails
the second test; the cumulative version passes."""

import pandas as pd

from feature_engineering import build_features


def _account_a(frame):
    out = frame[frame["Account"] == "A1"].sort_values("Timestamp")
    return out[["Timestamp", "account_txn_count_total", "txn_count_24h"]].reset_index(drop=True)


def test_account_count_is_cumulative_not_total(toy_transactions):
    df = build_features(toy_transactions)
    a = _account_a(df)
    # 5 transactions for A1 — past-only counts must run 1..5, NOT be 5 everywhere
    assert a["account_txn_count_total"].tolist() == [1, 2, 3, 4, 5]


def test_past_features_unchanged_when_future_changes(toy_transactions):
    base = _account_a(build_features(toy_transactions)).iloc[:3]

    mutated = toy_transactions.copy()
    # Change future amounts drastically and append extra future rows for A1
    mask = (mutated["Account"] == "A1") & (mutated["Timestamp"] >= "2022-09-01 03:00")
    mutated.loc[mask, "Amount Paid"] = 999999.0
    extra = mutated[mask].copy()
    extra["Timestamp"] = extra["Timestamp"] + pd.Timedelta(days=1)
    mutated = pd.concat([mutated, extra], ignore_index=True)

    after = _account_a(build_features(mutated)).iloc[:3]
    pd.testing.assert_frame_equal(base, after)

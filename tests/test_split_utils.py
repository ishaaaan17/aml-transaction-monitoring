"""The GNN/tabular split must be chronological: every train timestamp
must be <= every val timestamp <= every test timestamp. The old GNN split
sliced node positions and would fail these tests on shuffled input."""

import numpy as np
import pandas as pd

from split_utils import time_based_masks, time_split_bounds


def test_masks_are_chronological_on_shuffled_timestamps():
    ts = pd.date_range("2022-09-01", periods=100, freq="h")
    shuffled = ts.to_series().sample(frac=1.0, random_state=42).values  # node order != time order

    train_m, val_m, test_m = time_based_masks(shuffled)
    t = pd.to_datetime(shuffled)

    assert t[train_m].max() <= t[val_m].min()
    assert t[val_m].max() <= t[test_m].min()


def test_masks_partition_all_rows_with_expected_sizes():
    ts = pd.date_range("2022-09-01", periods=1000, freq="min")
    train_m, val_m, test_m = time_based_masks(ts)

    assert not (train_m & val_m).any()
    assert not (val_m & test_m).any()
    assert (train_m | val_m | test_m).all()
    assert abs(train_m.mean() - 0.70) < 0.02
    assert abs(test_m.mean() - 0.15) < 0.02


def test_bounds_match_tabular_quantile_rule():
    ts = pd.date_range("2022-09-01", periods=101, freq="h")
    train_end, val_end = time_split_bounds(ts)
    numeric = ts.values.astype("datetime64[ns]").astype(np.int64)
    assert train_end == np.quantile(numeric, 0.70)
    assert val_end == np.quantile(numeric, 0.85)

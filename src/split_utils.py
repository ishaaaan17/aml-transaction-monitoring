"""Torch-free helpers for the chronological (time-based) train/val/test split.

The tabular pipeline (train_tabular.prepare_data) splits by *timestamp
quantile*: train = timestamps <= 70th percentile, val = (70th, 85th],
test = > 85th. The GNN pipeline must use exactly the same rule on each
node's timestamp — NOT a positional slice of the node order (node order is
account order after feature engineering, so a positional slice is not a
time split at all; that was the bug this module fixes).

Everything here works on plain numpy/pandas so it can be unit-tested
without torch / torch_geometric installed.
"""

import numpy as np
import pandas as pd

TRAIN_Q = 0.70
VAL_Q = 0.85


def _to_numeric_time(timestamps):
    """Convert datetimes / datetime strings / numbers to int64 nanoseconds
    (or float for plain numeric input) so quantiles and comparisons work."""
    arr = np.asarray(timestamps)
    if arr.dtype.kind == "M":  # datetime64
        return arr.astype("datetime64[ns]").astype(np.int64)
    if arr.dtype.kind in "if":  # already numeric (e.g. stored int64 ns)
        return arr
    # strings / pandas Timestamps / object arrays
    return pd.to_datetime(pd.Series(arr)).values.astype("datetime64[ns]").astype(np.int64)


def time_split_bounds(timestamps, train_q=TRAIN_Q, val_q=VAL_Q):
    """Return (train_end, val_end) — the timestamp quantile cut points,
    identical in spirit to train_tabular.prepare_data()."""
    t = _to_numeric_time(timestamps)
    return np.quantile(t, train_q), np.quantile(t, val_q)


def time_based_masks(timestamps, train_q=TRAIN_Q, val_q=VAL_Q):
    """Return (train_mask, val_mask, test_mask) as numpy bool arrays.

    A row belongs to train iff ts <= train_end, to val iff
    train_end < ts <= val_end, and to test iff ts > val_end — the same
    boundary convention as the tabular split.
    """
    t = _to_numeric_time(timestamps)
    train_end, val_end = time_split_bounds(t, train_q, val_q)
    train_mask = t <= train_end
    val_mask = (t > train_end) & (t <= val_end)
    test_mask = t > val_end
    return train_mask, val_mask, test_mask

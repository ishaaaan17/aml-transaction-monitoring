import pandas as pd
import numpy as np


def add_transaction_level_features(df):
    """Features derivable from a single transaction row — no leakage risk."""
    df = df.copy()

    # Log-transform amounts (heavily right-skewed, confirmed in EDA)
    df["log_amount_paid"] = np.log1p(df["Amount Paid"])
    df["log_amount_received"] = np.log1p(df["Amount Received"])

    # Cross-bank flag — strong signal from EDA (98% laundering vs 86.4% normal)
    df["is_cross_bank"] = (df["From Bank"] != df["To Bank"]).astype("int8")

    # Currency mismatch flag
    df["currency_mismatch"] = (
        df["Payment Currency"] != df["Receiving Currency"]
    ).astype("int8")

    # Amount discrepancy between paid and received (should usually be ~0 unless FX)
    df["amount_diff"] = (df["Amount Paid"] - df["Amount Received"]).abs()
    df["log_amount_diff"] = np.log1p(df["amount_diff"])

    return df


def add_velocity_features(df, windows_hours=(24, 168)):
    """
    Rolling transaction count/volume per account over trailing windows.
    Keyed on (Account, From Bank) composite to correctly identify accounts —
    verified in EDA that Account alone is ~1:1 with (Account, From Bank),
    but composite key is the conceptually correct one regardless.
    """
    df = df.copy()
    df["account_key"] = df["Account"].astype(str) + "_" + df["From Bank"].astype(str)
    df = df.sort_values(["account_key", "Timestamp"]).reset_index(drop=True)

    for window_h in windows_hours:
        window = f"{window_h}h"
        count_col = f"txn_count_{window_h}h"
        sum_col = f"txn_volume_{window_h}h"

        grouped = df.set_index("Timestamp").groupby("account_key")

        rolling_count = grouped["Amount Paid"].rolling(window).count()
        rolling_sum = grouped["Amount Paid"].rolling(window).sum()

        df[count_col] = rolling_count.values
        df[sum_col] = rolling_sum.values

        df[count_col] = df[count_col] - 1
        df[sum_col] = df[sum_col] - df["Amount Paid"]

    return df


def add_account_frequency_features(df):
    """Degree-proxy: total transactions per (Account, From Bank)."""
    df = df.copy()
    if "account_key" not in df.columns:
        df["account_key"] = df["Account"].astype(str) + "_" + df["From Bank"].astype(str)
    account_counts = df["account_key"].value_counts()
    df["account_txn_count_total"] = df["account_key"].map(account_counts).astype("int32")
    return df


def build_features(df):
    """Full feature pipeline — run in order, each step depends on the previous."""
    df = add_transaction_level_features(df)
    df = add_velocity_features(df)
    df = add_account_frequency_features(df)
    return df


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_utils import load_transactions

    df = load_transactions(path="../data/HI-Small_Trans.csv")
    df = build_features(df)

    print(df.shape)
    print(df[[
        "log_amount_paid", "is_cross_bank", "currency_mismatch",
        "txn_count_24h", "txn_volume_24h", "account_txn_count_total", "Is Laundering"
    ]].head(10))

    # Sanity check: verify txn_count_24h only looks backward, not forward
    sample_account = df[df["account_txn_count_total"] > 3]["Account"].iloc[0]
    sample = df[df["Account"] == sample_account][["Timestamp", "txn_count_24h", "txn_volume_24h"]]
    print(sample)
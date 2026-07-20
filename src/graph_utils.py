import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from data_utils import load_transactions
from feature_engineering import build_features


def build_transaction_graph(df, feature_cols):
    """
    Build a temporal transaction-chain graph:
    - Each transaction is a node
    - Directed edges connect each transaction to the NEXT transaction
      chronologically on the same account (sender side and receiver side)
    """
    df = df.reset_index(drop=True)
    df["node_id"] = df.index

    edges = []

    df["sender_key"] = df["Account"].astype(str) + "_" + df["From Bank"].astype(str)
    sender_sorted = df.sort_values(["sender_key", "Timestamp"])
    sender_groups = sender_sorted.groupby("sender_key")["node_id"].apply(list)
    for node_list in sender_groups:
        if len(node_list) > 1:
            edges.extend(zip(node_list[:-1], node_list[1:]))

    df["receiver_key"] = df["Account.1"].astype(str) + "_" + df["To Bank"].astype(str)
    receiver_sorted = df.sort_values(["receiver_key", "Timestamp"])
    receiver_groups = receiver_sorted.groupby("receiver_key")["node_id"].apply(list)
    for node_list in receiver_groups:
        if len(node_list) > 1:
            edges.extend(zip(node_list[:-1], node_list[1:]))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    x = torch.tensor(df[feature_cols].astype(np.float32).values, dtype=torch.float32)
    y = torch.tensor(df["Is Laundering"].values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    print(f"Graph built: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Avg degree: {data.num_edges / data.num_nodes:.2f}")

    return data


if __name__ == "__main__":
    df = load_transactions(path="../data/HI-Small_Trans.csv")
    df = df[df["Timestamp"] <= "2022-09-10 23:59:59"]
    df = build_features(df)

    df = pd.get_dummies(df, columns=["Payment Format"], prefix="fmt")
    fmt_cols = [c for c in df.columns if c.startswith("fmt_")]

    feature_cols = [
        "log_amount_paid", "log_amount_received", "is_cross_bank",
        "currency_mismatch", "log_amount_diff",
        "txn_count_24h", "txn_volume_24h",
        "txn_count_168h", "txn_volume_168h",
        "account_txn_count_total",
    ] + fmt_cols

    df = df.dropna(subset=feature_cols)

    # Diagnostic: check dtypes before building the graph
    print(df[feature_cols].dtypes)
    print()

    data = build_transaction_graph(df, feature_cols)
    torch.save(data, "../models/transaction_graph.pt")
    print("Graph saved to ../models/transaction_graph.pt")
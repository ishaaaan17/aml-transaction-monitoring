import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

from train_gnn import GraphSAGE, get_time_based_masks
from data_utils import load_transactions
from feature_engineering import build_features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load("../models/transaction_graph.pt", weights_only=False)
    train_mask, val_mask, test_mask = get_time_based_masks(data, data.num_nodes)

    train_mean = data.x[train_mask].mean(dim=0, keepdim=True)
    train_std = data.x[train_mask].std(dim=0, keepdim=True)
    train_std[train_std == 0] = 1.0
    data.x = (data.x - train_mean) / train_std
    data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    test_loader = NeighborLoader(
        data, num_neighbors=[10, 5], batch_size=512,
        input_nodes=data.test_mask, shuffle=False,
    )

    model = GraphSAGE(in_channels=data.num_node_features).to(device)
    model.load_state_dict(torch.load("../models/graphsage_best.pt", weights_only=True))
    model.eval()

    all_probs, all_labels, all_node_ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch_gpu = batch.to(device)
            out = model(batch_gpu.x, batch_gpu.edge_index)
            probs = torch.softmax(out[:batch.batch_size], dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch.y[:batch.batch_size].cpu().numpy())
            all_node_ids.append(batch.n_id[:batch.batch_size].cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    node_ids = np.concatenate(all_node_ids)

    df = load_transactions(path="../data/HI-Small_Trans.csv")
    df = df[df["Timestamp"] <= "2022-09-10 23:59:59"]
    df = build_features(df)
    df["Payment Format Original"] = df["Payment Format"]  # keep readable copy before encoding
    df = pd.get_dummies(df, columns=["Payment Format"], prefix="fmt")
    fmt_cols = [c for c in df.columns if c.startswith("fmt_")]
    feature_cols = [
        "log_amount_paid", "log_amount_received", "is_cross_bank",
        "currency_mismatch", "log_amount_diff",
        "txn_count_24h", "txn_volume_24h",
        "txn_count_168h", "txn_volume_168h",
        "account_txn_count_total",
    ] + fmt_cols
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    df["node_id"] = df.index

    threshold = 0.1988  # from evaluate_gnn_test.py, ~90% recall operating point
    y_pred = (y_prob >= threshold).astype(int)

    results = pd.DataFrame({
        "node_id": node_ids, "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred
    })
    results = results.merge(df, on="node_id", how="left")

    false_negatives = results[(results["y_true"] == 1) & (results["y_pred"] == 0)]
    false_positives = results[(results["y_true"] == 0) & (results["y_pred"] == 1)].sort_values("y_prob", ascending=False)

    print(f"False Negatives (missed laundering): {len(false_negatives)}")
    print(false_negatives[["Timestamp", "Amount Paid", "Payment Format Original", "y_prob"]].head(10).to_string())
    print()
    print(f"Top False Positives (highest confidence wrong alerts): {len(false_positives)}")
    print(false_positives[["Timestamp", "Amount Paid", "Payment Format Original", "y_prob"]].head(10).to_string())

    false_negatives.to_csv("../docs/benchmarks/false_negatives_sample.csv", index=False)
    false_positives.head(50).to_csv("../docs/benchmarks/false_positives_sample.csv", index=False)
    print("\nSaved samples to docs/benchmarks/")
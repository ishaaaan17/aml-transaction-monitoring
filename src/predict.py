"""Inference: score transactions with a trained model.

Usage (run from src/):

    # XGBoost — scores any raw IBM-AML-format CSV
    python predict.py --model xgboost --input ../data/HI-Small_Trans.csv \
        --output ../results/predictions_xgboost.csv

    # GraphSAGE — transductive: scores the nodes of the built graph
    # (models/transaction_graph.pt), so it takes no --input CSV
    python predict.py --model graphsage \
        --output ../results/predictions_graphsage.csv

Outputs a CSV with a `y_prob` (laundering probability) column — plus
`y_true` when the input carries the `Is Laundering` label — ready to feed
into make_report.py.

If the required model artifact is missing this script exits with a clear
message saying which training step produces it. It never fabricates
predictions from an untrained model.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "models")

XGB_MODEL = os.path.join(MODELS_DIR, "xgboost_tuned.json")
GRAPH_FILE = os.path.join(MODELS_DIR, "transaction_graph.pt")
GNN_CHECKPOINT = os.path.join(MODELS_DIR, "graphsage_best.pt")


def _missing(path, produced_by):
    print(
        f"ERROR: required model artifact not found:\n  {path}\n"
        f"Produce it first by running (from src/):\n  python {produced_by}\n"
        "No predictions were generated.",
        file=sys.stderr,
    )
    sys.exit(2)


def predict_xgboost(input_csv, output_csv):
    if not os.path.exists(XGB_MODEL):
        _missing(XGB_MODEL, "train_tabular.py   # optionally tune_xgboost.py first")
    if not os.path.exists(input_csv):
        print(f"ERROR: input CSV not found: {input_csv}", file=sys.stderr)
        sys.exit(2)

    import pandas as pd
    import xgboost as xgb

    from data_utils import load_transactions
    from feature_engineering import build_features
    from train_tabular import FEATURE_COLS

    df = load_transactions(path=input_csv)
    df = build_features(df)
    df = pd.get_dummies(df, columns=["Payment Format"], prefix="fmt")
    fmt_cols = [c for c in df.columns if c.startswith("fmt_")]
    feature_cols = FEATURE_COLS + fmt_cols
    df = df.dropna(subset=FEATURE_COLS)

    model = xgb.XGBClassifier()
    model.load_model(XGB_MODEL)

    out = pd.DataFrame({"y_prob": model.predict_proba(df[feature_cols])[:, 1]})
    if "Is Laundering" in df.columns:
        out["y_true"] = df["Is Laundering"].values
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} predictions to {output_csv}")
    return output_csv


def predict_graphsage(output_csv):
    if not os.path.exists(GRAPH_FILE):
        _missing(GRAPH_FILE, "graph_utils.py")
    if not os.path.exists(GNN_CHECKPOINT):
        _missing(GNN_CHECKPOINT, "train_gnn.py")

    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.loader import NeighborLoader

    from train_gnn import GraphSAGE, get_time_based_masks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(GRAPH_FILE, weights_only=False)

    # Same chronological masks + train-only standardization as training.
    train_mask, val_mask, test_mask = get_time_based_masks(data)
    train_mean = data.x[train_mask].mean(dim=0, keepdim=True)
    train_std = data.x[train_mask].std(dim=0, keepdim=True)
    train_std[train_std == 0] = 1.0
    data.x = torch.nan_to_num((data.x - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)
    data.test_mask = test_mask

    loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=512,
                              input_nodes=data.test_mask, shuffle=False)
    model = GraphSAGE(in_channels=data.num_node_features).to(device)
    model.load_state_dict(torch.load(GNN_CHECKPOINT, weights_only=True))
    model.eval()

    probs, labels, node_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            probs.append(torch.softmax(out[:batch.batch_size], dim=1)[:, 1].cpu().numpy())
            labels.append(batch.y[:batch.batch_size].cpu().numpy())
            node_ids.append(batch.n_id[:batch.batch_size].cpu().numpy())

    out = pd.DataFrame({
        "node_id": np.concatenate(node_ids),
        "y_prob": np.concatenate(probs),
        "y_true": np.concatenate(labels),
    })
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} test-node predictions to {output_csv}")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score transactions with a trained AML model")
    parser.add_argument("--model", choices=["xgboost", "graphsage"], required=True)
    parser.add_argument("--input", help="Raw IBM-AML-format CSV (xgboost only)")
    parser.add_argument("--output", default="../results/predictions.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.model == "xgboost":
        if not args.input:
            parser.error("--model xgboost requires --input <csv>")
        predict_xgboost(args.input, args.output)
    else:
        predict_graphsage(args.output)

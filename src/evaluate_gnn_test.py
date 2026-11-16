import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch_geometric.loader import NeighborLoader

from train_gnn import GraphSAGE, get_time_based_masks, evaluate_gnn


import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix

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
        data,
        num_neighbors=[10, 5],
        batch_size=512,
        input_nodes=data.test_mask,
        shuffle=False,
    )

    model = GraphSAGE(in_channels=data.num_node_features).to(device)
    model.load_state_dict(torch.load("../models/graphsage_best.pt", weights_only=True))
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            probs = torch.softmax(out[:batch.batch_size], dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch.y[:batch.batch_size].cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    from sklearn.metrics import average_precision_score
    pr_auc = average_precision_score(y_true, y_prob)
    print(f"Test PR-AUC: {pr_auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    target_recall = 0.90
    idx = np.argmin(np.abs(recall - target_recall))
    threshold_at_recall = thresholds[min(idx, len(thresholds) - 1)]

    y_pred = (y_prob >= threshold_at_recall).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)

    print(f"At ~{recall[idx]*100:.1f}% recall (threshold={threshold_at_recall:.4f}):")
    print(f"  False Positive Rate: {fpr*100:.4f}%")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Save proof of this run: results/metrics.json + PR-curve plot
    # (previously every reported number was only a hand-typed markdown table).
    from make_report import report_from_predictions
    metrics, curve = report_from_predictions("graphsage", y_true, y_prob, results_dir="../results")
    print(f"\nSaved results bundle: ../results/metrics.json and {curve}")
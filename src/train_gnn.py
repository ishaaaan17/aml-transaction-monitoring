import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import average_precision_score
import numpy as np


class FocalLoss(nn.Module):
    """Focal loss with PER-CLASS alpha weighting, properly set for severe imbalance.
    alpha_pos should be high (close to 1) to upweight the rare positive class —
    NOT the object-detection-style alpha=0.25, which was wrong for our ~1243:1 ratio."""
    def __init__(self, alpha_pos=0.9, gamma=2.0):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = 1 - alpha_pos
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha_pos, self.alpha_neg)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()    


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_classes=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x


def get_time_based_masks(data, num_nodes):
    train_end = int(num_nodes * 0.70)
    val_end = int(num_nodes * 0.85)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    return train_mask, val_mask, test_mask


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch.batch_size

    return total_loss / loader.data.num_nodes


@torch.no_grad()
def evaluate_gnn(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        probs = F.softmax(out[:batch.batch_size], dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch.y[:batch.batch_size].cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    pr_auc = average_precision_score(all_labels, all_probs)
    return pr_auc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load("../models/transaction_graph.pt", weights_only=False)
    print(f"Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges")

    train_mask, val_mask, test_mask = get_time_based_masks(data, data.num_nodes)

    # Standardize features using TRAIN statistics only (avoid leakage from val/test)
    train_mean = data.x[train_mask].mean(dim=0, keepdim=True)
    train_std = data.x[train_mask].std(dim=0, keepdim=True)
    train_std[train_std == 0] = 1.0

    data.x = (data.x - train_mean) / train_std
    data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Feature stats after normalization - min: {data.x.min().item():.2f}, max: {data.x.max().item():.2f}")

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    print(f"Train positive rate: {data.y[train_mask].float().mean().item()*100:.4f}%")

    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=512,
        input_nodes=data.train_mask,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=512,
        input_nodes=data.val_mask,
        shuffle=False,
    )

    model = GraphSAGE(in_channels=data.num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = FocalLoss(alpha_pos=0.9, gamma=2.0)
    scaler = torch.cuda.amp.GradScaler()

    best_val_pr_auc = 0
    patience = 3
    epochs_without_improvement = 0
    max_epochs = 15

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_pr_auc = evaluate_gnn(model, val_loader, device)
        elapsed = time.time() - start
        print(f"Epoch {epoch}: loss={loss:.4f}, val PR-AUC={val_pr_auc:.4f}, time={elapsed:.1f}s")

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "../models/graphsage_best.pt")
            print(f"  -> New best, saved checkpoint")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  -> No improvement for {patience} epochs, stopping early")
                break

    print(f"\nBest val PR-AUC: {best_val_pr_auc:.4f}")
    print("Best model saved to ../models/graphsage_best.pt")
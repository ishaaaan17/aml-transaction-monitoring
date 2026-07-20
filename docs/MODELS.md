# Models

## 1. Rules-Based Baseline (src/rules_baseline.py)

Legacy-AML-system-style heuristic: score 0–3 based on (a) ACH payment format,
(b) cross-bank transaction, (c) amount in top 25th percentile — all three
chosen from validated EDA signals. Serves as the benchmark every ML model
must beat.

## 2. XGBoost (src/train_tabular.py)

Gradient-boosted trees on ~17 engineered features (transaction-level,
velocity, entity-level — see docs/PREPROCESSING.md). Chosen as the tabular
baseline because it handles mixed feature types well, trains fast on CPU
(no GPU contention with the GNN branch), and gives interpretable feature
importances. Class-weighted via `scale_pos_weight`. Hyperparameters tuned
via Optuna (40 trials) — see docs/EXPERIMENTS.md.

## 3. GraphSAGE (src/train_gnn.py)

Chosen over GAT specifically because it's cheaper to train (no attention
computation) — a deliberate choice given the 4GB VRAM constraint. Graph
structure: each transaction is a node; directed edges connect each
transaction to the next chronological transaction on the same account
(sender-side and receiver-side chains). This avoids the O(n²) blowup of a
full line graph, which would be infeasible given a hub account with 168K+
transactions (see docs/EDA.md). Trained with mini-batch neighbor sampling
(`NeighborLoader`, 2-hop, [10, 5] fanout) and mixed precision (`torch.cuda.amp`)
to fit VRAM constraints. Loss: focal loss with per-class alpha weighting
(alpha_pos=0.9), tuned specifically for the ~1243:1 imbalance — an early
attempt using object-detection-style alpha=0.25 failed badly (see
docs/EXPERIMENTS.md, Run 3 vs Run 4).

## Why Not a Fusion Model

Given time constraints, a combined tabular+graph fusion head was scoped but
not built — GraphSAGE alone already outperforms XGBoost by ~2x on PR-AUC,
and two clean standalone results were prioritized over a rushed fusion layer
that might underperform either component individually.

## Final Comparison

See docs/METRICS.md for the full results table. GraphSAGE is the
best-performing model (PR-AUC 0.1633 vs XGBoost's 0.0830 vs rules-based
0.0089).

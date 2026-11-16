# AML Transaction Monitoring — GraphSAGE (PyTorch Geometric) + XGBoost Benchmark Suite on the IBM AML Dataset

## System Overview

This project is an anti-money-laundering (AML) transaction-monitoring pipeline built on the IBM AML **HI-Small** synthetic dataset — 5,078,345 transactions over a 10-day window (Sept 1–10, 2022), of which just **0.1019%** (5,177) are laundering. At that imbalance, the interesting problem is not "can a model fit this data" but "can an evaluation be trusted": a model that flags nothing scores 99.9% accuracy, and a single leaky feature or a non-chronological split can manufacture results that look excellent and mean nothing.

So the pipeline is built as three models of increasing sophistication, all measured on **one shared, genuinely chronological test period** (Sept 9, 03:15 – Sept 10, 23:59; 761,422 transactions, 906 of them laundering):

1. A **rules-based baseline** — three threshold rules of the kind legacy AML systems actually run.
2. A **tabular XGBoost model** — gradient boosting over 17 engineered per-transaction features, tuned with a 40-trial Optuna search.
3. A **GraphSAGE graph neural network** — the same features, but each transaction also sees its position in the per-account transaction chains it belongs to.

Every number in this README was produced by the scripts in `src/` and, for the GNN, written automatically to `results/metrics.json` by the evaluation code — not typed into this document by hand.

## Results

| Model                  | PR-AUC | FPR @ 90% Recall |
| ---------------------- | ------ | ---------------- |
| Rules-based baseline   | 0.0072 | 30.76%           |
| XGBoost (Optuna-tuned) | 0.0638 | 11.58%           |
| GraphSAGE (GNN)        | 0.1994 | 11.16%           |

GraphSAGE delivers an **~27.7x improvement in PR-AUC** over the rules-based baseline (~3.1x over the tuned XGBoost model) and cuts the false positive rate by **~64% versus rules** (30.76% → 11.16%) at 90% recall. Against tuned XGBoost, its FPR at the 90%-recall operating point is similar (11.16% vs 11.58%) — the graph model's advantage is in ranking quality (PR-AUC) across all thresholds, which is what decides how much analyst time a fixed review budget buys.

(The rules baseline's integer score cannot hit exactly 90% recall; its FPR is at 89.1% recall.)

**Tuning, stated precisely:** XGBoost hyperparameters were tuned with a 40-trial Optuna search (`src/tune_xgboost.py`). GraphSAGE was tuned manually over learning rate and focal-loss alpha (Runs 3–4 in `docs/EXPERIMENTS.md`); it uses SAGEConv's default **mean aggregator — no aggregator comparison (mean/max/LSTM) and no Optuna search was run for the GNN**, and none is claimed.

See `docs/METRICS.md` for why PR-AUC (not accuracy/ROC-AUC) is the right metric here, and `docs/EVALUATION.md` for full evaluation details.

---

## 🚀 Core Architectural Features

### One Chronological Split, Shared by Every Model (`split_utils.py`)
Both pipelines split on **timestamp quantiles of the transaction stream**: train = timestamps up to the 70th percentile, validation = (70th, 85th], test = everything after the 85th percentile. In calendar terms:

| Split | Transactions | Window |
| ----- | ------------ | ------ |
| Train | 3,554,066 | Sept 1, 00:00 – Sept 7, 14:52 |
| Val   | 761,749   | Sept 7, 14:53 – Sept 9, 03:14 |
| Test  | 761,422   | Sept 9, 03:15 – Sept 10, 23:59 |

The tabular pipeline gets this by sorting on `Timestamp`. The GNN pipeline gets it from `split_utils.time_based_masks()`, applied to the per-node timestamps stored inside the graph file. This module exists because the GNN originally did something that *looked* equivalent and was not: it sliced node **positions** (first 70% / next 15% / last 15% of the node array). Node order in the graph is account order coming out of feature engineering, not time order — so the "time-based" split was no such thing, and every GNN number measured on it was meaningless. The graph builder now stores each node's timestamp (int64 nanoseconds) in the saved graph, and `train_gnn.py` **refuses to train** on an old graph file that lacks timestamps rather than silently falling back to the positional split.

### Leakage-Proof Feature Engineering (`feature_engineering.py`)
Seventeen features per transaction, in three groups, each designed so a row's features can only depend on that row and the past:

- **Transaction-level** (no temporal risk): `log_amount_paid`, `log_amount_received` (amounts are heavily right-skewed), `is_cross_bank` (98% of laundering vs 86.4% of normal transactions cross banks), `currency_mismatch`, `log_amount_diff`.
- **Velocity features**: per-account rolling count and volume over trailing **24h** and **168h** windows (`txn_count_24h`, `txn_volume_24h`, `txn_count_168h`, `txn_volume_168h`), keyed on the `(Account, From Bank)` composite. The current transaction is subtracted back out of its own window, so a transaction never counts itself.
- **Account frequency**: `account_txn_count_total` — despite the name, this is a **past-only cumulative count**: how many transactions this account had made *up to and including* this one, computed with a grouped `cumcount()` in timestamp order.

That last feature is the second bug this codebase carries a scar from. It was originally a `value_counts()` over the *entire* dataframe — every row encoded its account's **future** transaction count, including test-period activity, and since feature building runs before the split, the leak reached all three splits. The fix is pinned down by a unit test (`tests/test_features_no_future.py`) that truncates the dataframe and asserts earlier rows' features do not change.

The remaining seven features are one-hot payment-format flags (`fmt_ACH`, `fmt_Bitcoin`, `fmt_Cash`, `fmt_Cheque`, `fmt_Credit Card`, `fmt_Reinvestment`, `fmt_Wire`).

### Rules-Based Baseline (`rules_baseline.py`)
Three rules drawn from the EDA (`docs/EDA.md`), scored 0–3 and used as a ranking score:

- Payment format is **ACH** (86.6% of laundering vs 11.8% of normal)
- Transaction is **cross-bank**
- Amount is in the **top quartile**

One detail matters here: the amount threshold is the 75th percentile of the **train period only**. The original script derived it from the test set it was evaluating on — a small, easy-to-miss leak that flattered the baseline. With the threshold fixed honestly, the baseline lands at PR-AUC **0.0072** with a 30.76% false positive rate near 90% recall: it catches most laundering, at the cost of flagging nearly a third of all legitimate traffic. That is the operational problem the learned models exist to solve.

### Tabular Model — XGBoost + Optuna (`train_tabular.py`, `tune_xgboost.py`)
XGBoost over the same 17 features, with the imbalance handled by `scale_pos_weight` computed from the train split (≈ **1243.42** — the actual negative:positive ratio, not a hand-picked constant). Training optimises `aucpr` directly, with early stopping (20 rounds) on the validation split.

Hyperparameters come from a 40-trial Optuna search maximising validation PR-AUC (`models/best_xgb_params.json`). The winning trial (33/40, val PR-AUC 0.0799) chose depth 9, learning rate 0.124, 314 trees, subsample 0.621, colsample 0.520, gamma 2.385. On the test period the tuned model scores PR-AUC **0.0638**, FPR **11.58%** at 90% recall — tuning alone was worth ~60% over the default hyperparameters (0.0400) on the identical split.

### Transaction-Chain Graph (`graph_utils.py`)
The graph's nodes are **transactions, not accounts** — 5,077,237 of them. A directed edge links each transaction to the **next chronological transaction on the same account**, built twice over: once along the sender side (`Account` + `From Bank`) and once along the receiver side (`Account.1` + `To Bank`). Total: **9,236,876 edges**, average degree 1.82.

The obvious alternative — an account graph where an edge means "these accounts transacted" — was rejected on arithmetic: the largest hub account has 168,672 transactions, and connecting counterparties pairwise through hubs explodes towards ~14 billion edges. The chain graph keeps edges roughly linear in transaction count while preserving exactly the structure laundering typologies live in: sequences. Layering and structuring are *patterns over consecutive transactions*, and a chain lets a 2-layer GNN see a transaction's predecessor and successor context on both sides of the flow. The full reasoning is in `docs/DESIGN_DECISIONS.md`.

Node features are the same 17 as the tabular model, standardised using **train-split statistics only**. Each node also carries its timestamp (see the split section above) and its label.

### GraphSAGE Model (`train_gnn.py`)
A compact two-layer architecture, deliberately sized for a 4GB GPU:

```
17 ──► SAGEConv ──► 64 ──► ReLU ──► Dropout(0.3) ──► SAGEConv ──► 64 ──► ReLU ──► Linear ──► 2
```

- **Sampling:** `NeighborLoader` with fan-out **[10, 5]**, batch size 512 — the full graph never materialises on the GPU.
- **Loss:** focal loss with per-class alpha (`alpha_pos=0.9`, `gamma=2.0`). The alpha is set *for* the imbalance — the object-detection convention of alpha=0.25 would down-weight the very class that is 1,243x rarer.
- **Optimiser:** Adam (lr 0.001, weight decay 1e-4), mixed precision (AMP), gradient clipping at 1.0.
- **Schedule:** max 15 epochs, early stopping with patience 3 on validation PR-AUC.

In the reference run, the best checkpoint came from **epoch 1** (val PR-AUC **0.2024**), training stopped after epoch 4, and the saved checkpoint scores **0.1994** PR-AUC on the test period. Best-epoch-1 is unusual and worth stating plainly: with focal loss at this imbalance, the model reaches its best ranking early and then overfits the majority pattern; the early-stopping checkpoint is what makes that a non-issue.

### Evaluation Methodology (`evaluate_gnn_test.py`, `docs/METRICS.md`)
Two metrics, chosen for a 0.1%-positive problem:

- **PR-AUC** via `sklearn.metrics.average_precision_score`. *Not* `auc(recall, precision)` — trapezoidal interpolation is invalid on PR curves and, in this project's own history, once made the 3-rule baseline look 3.7x better than it was (see `docs/errors/KNOWN_ISSUES.md`).
- **FPR at 90% recall** — the operating number a compliance team actually feels: fix the catch rate at 9 out of 10 laundering transactions, then count what fraction of legitimate traffic gets flagged for human review.

Accuracy is not reported because it is meaningless here (predict "legitimate" everywhere: 99.8981% accurate, zero laundering caught), and ROC-AUC is not reported because at ~1000:1 imbalance it stays flattering long after precision has collapsed.

The GNN evaluation writes its own proof bundle — `results/metrics.json` (PR-AUC, threshold, full confusion matrix: TP=815, FP=84,867, TN=675,649, FN=91) and `results/pr_curve_graphsage.png`. `src/make_report.py` regenerates reported figures from prediction files, and `src/predict.py` scores new data with either saved model (`--model xgboost` on a raw CSV, `--model graphsage` on the built graph).

### Failure Analysis (`extract_failure_cases.py`)
At the default 0.5 threshold the GraphSAGE model leaves **171 false negatives** and **44,055 false positives** on the test period (samples in `docs/benchmarks/`). The patterns are legible:

- The missed laundering is concentrated in **Cheque, Cash and Credit Card** formats, with scores clustered at 0.11–0.16 — the model has learned to associate laundering primarily with the ACH channel (where most of it is) and under-prices the rest.
- The loudest false positives are almost all **ACH** transactions scoring just over 0.5 — the mirror image of the same bias.

Both are documented rather than hidden, because they point at the honest next step (per-format calibration), not at a number to round away.

### One Disclosed Limitation of the GNN Setup
The chain graph is built over the full 10-day window and split by node masks, so some chain edges **cross the split boundary**, and neighbour sampling is therefore *transductive*: a train node's embedding can aggregate a neighbour that sits in a later period. The node **features** are strictly past-only — no future information enters through them — but the graph *structure* is not causally masked. This is standard for transductive GNN evaluation and is disclosed in `docs/DESIGN_DECISIONS.md` rather than buried; the GNN result should be read as a transductive result.

---

## 🛠️ System Architecture Diagram

```
                        HI-Small_Trans.csv  (5,078,345 txns, Sept 1-10 2022)
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │  feature_engineering.py        │
                        │  17 features, past-only        │
                        │  (velocity windows, cumulative │
                        │   counts, format one-hots)     │
                        └───────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              │                         │                          │
              ▼                         ▼                          ▼
   ┌───────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
   │ rules_baseline.py │   │ train_tabular.py        │   │ graph_utils.py          │
   │ 3 rules, score 0-3│   │ XGBoost                 │   │ transaction-chain graph │
   │ threshold from    │   │ scale_pos_weight=1243   │   │ 5,077,237 nodes         │
   │ TRAIN period only │   │ Optuna 40 trials        │   │ 9,236,876 chain edges   │
   └─────────┬─────────┘   └───────────┬────────────┘   └───────────┬────────────┘
             │                         │                          │
             │                         │                          ▼
             │                         │               ┌────────────────────────┐
             │                         │               │ train_gnn.py            │
             │                         │               │ 2x SAGEConv (64), focal │
             │                         │               │ loss, NeighborLoader    │
             │                         │               │ [10,5], AMP, early stop │
             │                         │               └───────────┬────────────┘
             │                         │                          │
             ▼                         ▼                          ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │        SHARED chronological test period (split_utils: 70/85%             │
   │        timestamp quantiles) — Sept 9 03:15 to Sept 10 23:59              │
   │        761,422 txns · 906 laundering                                     │
   │                                                                          │
   │   Rules 0.0072 / 30.76%  ·  XGBoost 0.0638 / 11.58%  ·                   │
   │   GraphSAGE 0.1994 / 11.16%        (PR-AUC / FPR @ 90% recall)          │
   └──────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
              results/metrics.json · pr_curve_graphsage.png · docs/benchmarks/
```

---

## 📦 Project Directory Structure

```
aml-transaction-monitoring/
├── src/
│   ├── data_utils.py            # CSV loading
│   ├── feature_engineering.py   # 17 features — velocity windows, past-only counts
│   ├── split_utils.py           # Chronological split (timestamp quantiles), torch-free
│   ├── rules_baseline.py        # 3-rule legacy-style baseline
│   ├── tune_xgboost.py          # 40-trial Optuna search -> models/best_xgb_params.json
│   ├── train_tabular.py         # XGBoost train + test evaluation
│   ├── graph_utils.py           # Builds the transaction-chain graph (+ timestamps)
│   ├── train_gnn.py             # GraphSAGE: focal loss, NeighborLoader, early stopping
│   ├── evaluate_gnn_test.py     # Final GNN test eval -> results/metrics.json + PR curve
│   ├── extract_failure_cases.py # FN/FP sampling -> docs/benchmarks/
│   ├── predict.py               # Score new data with either saved model
│   ├── make_report.py           # Regenerates reported metrics from prediction files
│   ├── diagnose_model.py        # Model diagnostics
│   └── diagnose_time_shift.py   # Train/test distribution-shift check
│
├── tests/
│   ├── test_split_utils.py          # Split is chronological, boundaries correct
│   ├── test_features_no_future.py   # Features unchanged when future rows are cut
│   └── test_rules_and_report.py     # Baseline + report generation
│
├── docs/
│   ├── DATA.md                  # Dataset download and placement
│   ├── EDA.md                   # Exploration findings the rules/features came from
│   ├── METRICS.md               # Why PR-AUC + FPR@recall, full results
│   ├── EVALUATION.md            # Evaluation protocol in detail
│   ├── MODELS.md                # Model cards
│   ├── DESIGN_DECISIONS.md      # Chain graph, cross-split edges, fusion decision
│   ├── EXPERIMENTS.md           # Run log (incl. GraphSAGE manual tuning runs)
│   ├── PREPROCESSING.md         # Preprocessing notes
│   ├── benchmarks/              # Failure-case samples (FN/FP CSVs)
│   └── errors/
│       ├── KNOWN_ISSUES.md      # Bugs found & fixed (split, leakage, PR-AUC)
│       └── FAILURE_CASES.md     # Failure analysis write-up
│
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory analysis
│
├── results/
│   ├── metrics.json             # Auto-written by the evaluation code (proof bundle)
│   ├── pr_curve_graphsage.png   # PR curve of the final GNN run
│   └── README.md                # What a full re-run generates here
│
├── data/                        # HI-Small_Trans.csv goes here (gitignored)
├── models/                      # Saved models + graph file (gitignored)
├── requirements.txt             # Read the header — PyTorch/PyG need matched wheels
├── .github/workflows/ci.yml     # Test suite on push
└── LICENSE                      # MIT
```

---

## ⚡ Full Execution & Deployment Guide — Complete Pipeline Re-Run

This walks through reproducing every number in the Results table from a clean checkout.

### 1. Get the dataset

Download `HI-Small_Trans.csv` from the Kaggle dataset `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` (IBM Transactions for AML) and place it in `data/`. Nothing runs without it; the file is ~5.08M rows. All scripts use only the Sept 1–10 window (the raw file's background traffic collapses after day 10 — see `docs/errors/KNOWN_ISSUES.md`).

### 2. Set up the environment

Requires Python 3.13 and, for the GNN, an NVIDIA GPU (developed and run on an **RTX 3050, 4GB VRAM**).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Read the header of `requirements.txt` before installing.** The pinned CUDA build of PyTorch is not on PyPI — it must come from the PyTorch wheel index, and the PyTorch Geometric extensions (`pyg-lib`, `torch-scatter`, `torch-sparse`) must match your exact torch + CUDA combination, installed from the PyG wheel index. The header contains the exact commands for the reference build (torch 2.12.1 + cu126). Verify with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. Run the pipeline (from `src/`)

```bash
cd src

# Rules baseline (~1 min)
python rules_baseline.py

# Tabular model — tune first, then train (~12 min tuning + ~2 min training)
python tune_xgboost.py        # 40 Optuna trials -> ../models/best_xgb_params.json
python train_tabular.py       # uses tuned params if present, else defaults

# Graph + GNN (~5 min graph build, ~5-15 min training)
python graph_utils.py         # -> ../models/transaction_graph.pt (with timestamps)
python train_gnn.py           # -> ../models/graphsage_best.pt
python evaluate_gnn_test.py   # -> ../results/metrics.json + PR curve

# Failure analysis + inference (optional)
python extract_failure_cases.py
python predict.py --model xgboost --input ../data/HI-Small_Trans.csv --output ../results/predictions_xgboost.csv
```

Order matters in two places: `tune_xgboost.py` must precede `train_tabular.py` or the tabular model silently falls back to default hyperparameters (it prints which it used — check for `Using tuned params`), and `graph_utils.py` must be re-run if you ever pull a code update that changes the graph format, because `train_gnn.py` rejects graph files without per-node timestamps by design.

### 4. What to expect

- `rules_baseline.py` prints `PR-AUC: 0.0072` and, at ~89.1% recall, `FPR: 30.76%` (TP=807, FP=233,912, FN=99).
- `train_tabular.py` prints the split windows (train ends Sept 7, 14:52; test starts Sept 9, 03:15), `scale_pos_weight: 1243.42`, and with tuned params finishes at `PR-AUC: 0.0638`, `FPR: 11.58%` at 90% recall.
- `graph_utils.py` reports `5,077,237 nodes, 9,236,876 edges, avg degree 1.82`.
- `train_gnn.py` prints one line per epoch (`Epoch 1: loss=..., val PR-AUC=0.2024`) and stops early when validation stalls; the reference run stopped after epoch 4 with the epoch-1 checkpoint kept.
- `evaluate_gnn_test.py` prints the headline: `Test PR-AUC: 0.1994`, `FPR: 11.16%` at 90% recall (threshold 0.1606; TP=815, FP=84,867, TN=675,649, FN=91) — and writes the same numbers to `results/metrics.json`, so the README table can be audited against the file.

### 5. Run the tests

```bash
python -m pytest tests/ -q
```

Eight tests cover the three properties this project has been bitten by: the split is genuinely chronological, features do not change when future rows are removed, and the rules baseline + report generation behave as documented. CI (`.github/workflows/ci.yml`) runs the same suite on every push.

---

## Known Limitations

- **HI-Small only.** The larger IBM AML variants (HI-Medium / HI-Large) are untested — the 4GB VRAM budget and neighbour-sampling costs were sized around HI-Small.
- **10 of 17 days.** Only Sept 1–10 of the dataset window is used, due to a background-traffic collapse artifact in the raw data after day 10 (documented in `docs/errors/KNOWN_ISSUES.md`).
- **Transductive graph evaluation.** Chain edges cross the split boundary; features are past-only, graph structure is not causally masked (see the disclosure above and `docs/DESIGN_DECISIONS.md`).
- **No fusion model.** A tabular+GNN ensemble was considered and time-boxed out (see `docs/DESIGN_DECISIONS.md`).
- **GraphSAGE tuning is manual.** Learning rate and focal-loss alpha were tuned over a handful of runs; the aggregator was never compared (mean is the SAGEConv default). A systematic GNN search is the clearest headroom left in the project.

---

## License

MIT License — see [`LICENSE`](./LICENSE) for the full text.

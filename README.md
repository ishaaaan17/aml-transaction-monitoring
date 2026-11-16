# AML Transaction Monitoring System

A transaction-monitoring pipeline that detects likely money-laundering activity
using a tabular gradient-boosted model and a graph neural network, benchmarked
against a rules-based baseline. Built on the IBM AML (HI-Small) synthetic dataset.

## Results

| Model                  | PR-AUC | FPR @ 90% Recall |
| ---------------------- | ------ | ---------------- |
| Rules-based baseline   | 0.0089 | 33.28%           |
| XGBoost (Optuna-tuned) | 0.0830 | 11.38%           |
| GraphSAGE (GNN)        | 0.1633 | 6.58%            |

GraphSAGE delivers an ~18.3x improvement in PR-AUC and an 80% reduction in
false positive rate versus the rules-based baseline, at matched 90% recall.

> ⚠️ **Re-run pending (2026-11-17 fix):** these GraphSAGE numbers were
> measured with a train/val/test split that turned out to slice node
> *positions* (account order), not timestamps — and one feature
> (`account_txn_count_total`) leaked future counts. Both bugs are now
> fixed in code (chronological split via `src/split_utils.py`, past-only
> cumulative feature), but the model has **not yet been re-trained** on
> the corrected pipeline, so the GraphSAGE row above will change when it
> is. The XGBoost and rules rows were measured on a genuinely
> chronological split and are unaffected. Updated numbers will be backed
> by `results/metrics.json` (written automatically by the evaluation
> code) instead of hand-typed tables.
>
> **Tuning, stated precisely:** XGBoost hyperparameters were tuned with a
> 40-trial Optuna search (`src/tune_xgboost.py`). GraphSAGE was tuned
> manually over learning rate and focal-loss alpha (Runs 3–4 in
> `docs/EXPERIMENTS.md`); it uses SAGEConv's default **mean aggregator —
> no aggregator comparison (mean/max/LSTM) and no Optuna search was run
> for the GNN**, and none is claimed.

See `docs/METRICS.md` for why PR-AUC (not accuracy/ROC-AUC) is the right
metric here, and `docs/EVALUATION.md` for full evaluation details.

## Setup

Requires Python 3.13, an NVIDIA GPU with CUDA 12.6-compatible drivers
(tested on RTX 3050, 4GB VRAM), and the HI-Small dataset from Kaggle
(`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`) — download
`HI-Small_Trans.csv` and place it in `data/`.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

PyTorch Geometric extensions require separate installation matched to your
torch/CUDA version — see comments in `requirements.txt` and
`docs/DESIGN_DECISIONS.md` for the exact commands used.

## Running the Pipeline

```bash
cd src

# 1. Explore the data (optional — see notebooks/01_eda.ipynb for full EDA)
python data_utils.py

# 2. Train and evaluate the rules-based baseline
python rules_baseline.py

# 3. Train and evaluate the tabular model (XGBoost)
python tune_xgboost.py      # Optuna hyperparameter search (~30-45 min)
python train_tabular.py     # trains final model with tuned params

# 4. Build the transaction graph and train the GNN
python graph_utils.py       # builds and saves the graph (~few minutes)
python train_gnn.py         # trains GraphSAGE (~10-15 min with early stopping)
python evaluate_gnn_test.py # final test-set evaluation
```

## Project Structure

Key docs: `docs/DATA.md` (dataset download and placement — read this
first), `docs/EDA.md` (data exploration and findings), `docs/METRICS.md`
(evaluation methodology), `docs/errors/KNOWN_ISSUES.md` (bugs found and
fixed during development), `docs/DESIGN_DECISIONS.md` (why each modeling
choice was made, including how cross-split graph edges are handled).
After training, `src/predict.py` scores new transactions with a saved
model, and evaluation writes its proof bundle to `results/` (see
`results/README.md`).

## Known Limitations

- Trained on HI-Small only; larger IBM AML variants (HI-Medium/HI-Large)
  not yet tested due to 4GB VRAM constraint
- No tabular+GNN fusion model built (time-boxed decision, see
  docs/DESIGN_DECISIONS.md)
- Data restricted to Sept 1–10 of the 17-day window due to a background
  traffic collapse artifact in the raw dataset after day 10 (see
  docs/errors/KNOWN_ISSUES.md)

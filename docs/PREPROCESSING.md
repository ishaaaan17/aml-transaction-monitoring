# Preprocessing & Feature Engineering

## Pipeline (src/feature_engineering.py)

### Transaction-level features

- `log_amount_paid`, `log_amount_received`: log1p transform — raw amounts
  are heavily right-skewed (max values in the trillions, see docs/EDA.md)
- `is_cross_bank`: binary flag, From Bank != To Bank
- `currency_mismatch`: binary flag, Payment Currency != Receiving Currency
- `log_amount_diff`: log1p of |Amount Paid - Amount Received|

### Velocity features (leakage-critical)

- `txn_count_24h`, `txn_volume_24h`, `txn_count_168h`, `txn_volume_168h`:
  rolling trailing-window count/sum per account
- Computed via `groupby + rolling` on a composite `(Account, From Bank)` key
  — NOT `Account` alone, since a small number of accounts (4 out of ~497K)
  are ambiguous without the bank qualifier (see docs/EDA.md)
- Each row's own transaction is explicitly excluded from its own rolling
  window (subtracted out after computing the window) to prevent the current
  transaction leaking into its own "history" features

### Entity-level features

- `account_txn_count_total`: total transaction count per (Account, From Bank)
  — simple degree-proxy; full graph centrality deferred to the GNN branch

### Categorical Encoding

- `Payment Format` one-hot encoded (7 categories) — chosen over target
  encoding to avoid any leakage risk and because tree-based models handle
  one-hot splits natively well

### Excluded Features

- Raw hour-of-day: confirmed confounded with Payment Format (Reinvestment
  transactions batch-timestamped near midnight) — see docs/errors/KNOWN_ISSUES.md

## GNN-Specific Preprocessing

- Node features standardized (zero mean, unit variance) using TRAIN-set
  statistics only, applied to val/test — required both for training
  stability (raw feature magnitudes up to 8.8e8 overflow float16 during
  mixed-precision training) and general good practice for neural nets
- `torch.nan_to_num` applied as a safety net after standardization

## Class Imbalance Handling

- XGBoost: `scale_pos_weight` (ratio of negative to positive class counts)
- GraphSAGE: focal loss with `alpha_pos=0.9`, `gamma=2.0` — tuned specifically
  for this dataset's ~1243:1 imbalance (see docs/EXPERIMENTS.md)

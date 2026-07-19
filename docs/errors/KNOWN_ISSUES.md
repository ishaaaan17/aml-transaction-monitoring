# Known Issues

## Hour-of-Day Confounded with Payment Format

Raw transaction hour appeared to show a laundering-relevant pattern (spike at hour 0),
but this is fully explained by Reinvestment-type transactions being batch-timestamped
near midnight in the synthetic data generator — not a genuine behavioral signal.
Verified via crosstab of hour vs payment format (see docs/EDA.md).
Do not use raw hour-of-day as a model feature without correcting for this confound.

## Data Artifact: Background Traffic Collapse After Sept 10

Daily transaction volume drops from 200K-1.1M/day (Sept 1-10) to under 400/day
(Sept 11-18), while laundering rate in that sparse tail jumps to 56-73%.
This indicates the simulator's normal background traffic generation stops
around day 10, leaving only trailing transactions from long-running laundering
scenarios still being logged. These days are not representative of real daily
activity and were excluded from all train/val/test splits (capped at Sept 10).
Verified via daily groupby breakdown — see src/diagnose_time_shift.py.

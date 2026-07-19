# EDA Findings — HI-Small Dataset

## Dataset Overview

- 5,078,345 transactions, 17-day window (2022-09-01 to 2022-09-18)
- 11 columns: Timestamp, From Bank, Account, To Bank, Account.1, Amount Received,
  Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering

## Class Imbalance

- 5,177 laundering transactions out of 5,078,345 total (0.1019%)
- ~980:1 imbalance ratio — accuracy is not a usable metric here.
  PR-AUC and false-positive-rate-at-fixed-recall are the metrics that matter.

## Key Signals Found

### Payment Format (strongest single signal)

- ACH used in 86.6% of laundering transactions vs 11.8% of normal transactions
- Payment Format will be one of the most important categorical features

### Transaction Amount

- Laundering median (~8,667) is ~6x higher than normal median (~1,411)
- Both distributions are heavily right-skewed (max values in the trillions) —
  log-transform needed before feeding to models, especially linear/distance-based ones

### Cross-Bank Activity

- Laundering transactions are cross-bank 98.0% of the time vs 86.4% for normal
- Consistent with layering typology (moving funds across institutions)

## Known Pitfall: Hour-of-Day Confound

- Raw hour-of-day initially looked like a signal (spike at hour 0), but this is
  a data-generation artifact: 37.2% of hour-0 transactions are "Reinvestment" format,
  vs ~5.5% at every other hour. Reinvestment transactions are batch-timestamped near
  midnight in this synthetic data.
- Conclusion: raw hour-of-day is confounded with Payment Format and should NOT be
  used as an independent feature as-is. Logged in docs/errors/KNOWN_ISSUES.md.

## Currency / Format Distribution

- Dominant currencies: US Dollar (1.9M), Euro (1.17M), others long-tail
- Dominant payment formats: Cheque (1.86M), Credit Card (1.32M), ACH (601K)

## Implications for Feature Engineering

- Payment Format: one-hot or target-encode — highest priority categorical feature
- Amount Paid/Received: log-transform (log1p) before use
- Cross-bank flag (From Bank != To Bank): binary feature, clear signal
- Hour-of-day: exclude raw, or engineer relative to payment-format baseline if used at all

## Hub Account Structure

Account 100428660 (From Bank 70) processes 168,672 transactions across the 17-day
window — ~3.3% of all transactions in the dataset. Verified this is legitimate
(single consistent From Bank, timestamps spread naturally across the full date range,
not a data bug). This account carries 243 laundering transactions (vs 5,177 total),
suggesting hub-and-spoke laundering patterns — relevant signal for the graph/GNN branch,
where degree centrality and hub-adjacency should be strong features.

# Failure Cases — GraphSAGE Model

## Summary

85 false negatives (missed laundering) and 50,030 false positives at the
90%-recall operating threshold (0.1988). Both failure types are dominated
by ACH-format transactions — a specific, diagnosable pattern rather than
random noise.

## False Negatives (Missed Laundering) — the costly failure mode

All 10 sampled false negatives are ACH-format transactions with predicted
probabilities clustered tightly between 0.16 and 0.20 — just under the
0.1988 decision threshold. Amounts vary widely ($273 to $28,114), and
timestamps are spread across different days (not clustered in time),
ruling out an obvious amount- or timing-based explanation.

**Interpretation**: the model has learned that ACH format correlates
strongly with laundering (consistent with EDA: ACH used in 86.6% of
laundering vs 11.8% of normal transactions, and XGBoost's feature
importance showing fmt_ACH at 0.82). But within the ACH subgroup, the
model isn't finding a strong secondary signal to separate true positives
from true negatives — probabilities for actual laundering cases land just
below threshold rather than being pushed clearly above it.

**What this means practically**: this is not a "model is broken" failure
— it's a "model needs a stronger secondary signal for the ACH subgroup"
failure. A natural next step (not yet implemented, noted in docs/TODO.md)
would be an ACH-specific feature — e.g., interaction terms between ACH
format and velocity features, since layering-style laundering going through
ACH may show more account-level velocity signal that isn't currently being
weighted heavily enough for this specific subgroup.

## False Positives (False Alerts)

Top-confidence false positives (probability 0.58–0.62) are also
exclusively ACH. Several show large amounts ($218,626, $259,599) alongside
smaller ones ($2,812, $3,819) — no consistent amount pattern distinguishing
them from true positives. Several cluster right around midnight on Sept 1
(00:12–00:27) — this overlaps with the known Reinvestment-format
batch-timestamping artifact documented in docs/errors/KNOWN_ISSUES.md,
though these specific transactions are ACH, not Reinvestment, so it may be
coincidental rather than the same root cause. Worth a follow-up check.

## Cost Asymmetry

In an AML system, false negatives (missed laundering) are far more costly
than false positives (analyst reviews a normal transaction) — regulatory
and reputational risk from missed laundering vastly outweighs alert-review
labor cost. The 90%-recall operating point was deliberately chosen to bias
toward catching more true positives at the cost of a higher false-positive
volume, consistent with this asymmetry. See docs/METRICS.md for the
recall/FPR tradeoff at this operating point.

## Sample Data

Full false negative list: docs/benchmarks/false_negatives_sample.csv
Top 50 false positives: docs/benchmarks/false_positives_sample.csv

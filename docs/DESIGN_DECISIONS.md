# Design Decisions

## Why HI-Small over other IBM AML variants

4GB VRAM and time constraints ruled out Medium/Large variants for this phase.
HI (Higher Illicit ratio) chosen over LI to get more positive examples to
learn from without needing a larger dataset overall.

## Why time-based split, not random

Velocity features depend on transaction history — a random split would let
future transactions leak into a training row's "past" window. Time-based
split (70/15/15 by timestamp quantile) preserves temporal integrity.

## Why data restricted to Sept 1–10

See docs/errors/KNOWN_ISSUES.md — background traffic collapses after day 10,
producing a non-representative tail that would corrupt any split including it.

## Why PR-AUC via average_precision_score, not auc(recall, precision)

`auc()` linearly interpolates between PR curve points, which is invalid for
PR curves and overestimates performance — this bug initially made a 4-value
rules-based score look 3.7x stronger than it really was. Documented in
docs/errors/KNOWN_ISSUES.md and docs/METRICS.md.

## Why GraphSAGE over GAT

GAT's attention mechanism is more expensive to compute — not a good fit for
4GB VRAM under time pressure. GraphSAGE gives strong results (see docs/METRICS.md)
without that overhead.

## Why a transaction-chain graph, not a full line graph

A full pairwise line graph (connecting every transaction pair sharing an
account) would create ~14 billion edges from the single largest hub account
alone (168,672 transactions). Infeasible. Chain graph (each transaction
linked only to its chronological neighbor per account, sender and receiver
side) keeps edges roughly linear in transaction count (~1.82 avg degree)
while still capturing the sequential nature of laundering typologies
(layering, structuring) — arguably more correct for this domain than an
undirected all-pairs graph would have been anyway.

## Why no fusion model (tabular + GNN combined)

Time-boxed decision — two clean standalone results were prioritized over a
rushed fusion layer. GraphSAGE alone already substantially outperforms
XGBoost, so fusion was deprioritized rather than attempted under time pressure.

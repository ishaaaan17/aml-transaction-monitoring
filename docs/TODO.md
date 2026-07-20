# TODO

## High Priority (if time allows)

- [ ] Investigate ACH-subgroup discrimination weakness (see
      docs/errors/FAILURE_CASES.md) — try ACH-specific interaction features
      (e.g., ACH x velocity) to push false negatives further from threshold
- [ ] Check whether the false-positive midnight clustering (Sept 1,
      00:12-00:27) shares a root cause with the known Reinvestment
      batch-timestamping artifact, or is coincidental

## Medium Priority

- [ ] Tabular + GNN fusion model (deprioritized for time — see
      docs/DESIGN_DECISIONS.md)
- [ ] Test on larger IBM AML variants (HI-Medium/HI-Large) once access to
      stronger GPU (college machine)
- [ ] SHAP analysis on XGBoost model for interpretability (planned in
      original project spec, not yet done)

## Low Priority / Nice to Have

- [ ] Network visualization (Plotly) of a few flagged laundering rings
- [ ] Try GAT instead of GraphSAGE now that VRAM/time pressure may ease
      on stronger hardware, to see if attention improves ACH-subgroup
      discrimination specifically
- [ ] Extend data range beyond Sept 10 if a way is found to filter out
      only the degenerate low-volume days rather than dropping the whole tail

## Known Issues Carried Forward

See docs/errors/KNOWN_ISSUES.md for full list of bugs found and fixed
during development (hour-of-day confound, Sept-10 data collapse, PR-AUC
interpolation bug, float16 overflow in GNN training, focal loss alpha
misconfiguration).

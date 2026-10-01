# Figures, tables and reproducibility requirements

> Historical design/preparation record. Current execution status, corrections and
> evidence are maintained in [the hardening record](16_hardening_execution.md)
> and [the revised report](../results/settlement_study_v2_1/REPORT.md).

## Most useful publication displays

| Display | Source / design | Claim it can support |
|---|---|---|
| F1 origin/availability timeline | One actual event; show q observations, selected origin, publication and settlement | Reader understands the information set |
| F2 two-panel lead-time curve | Common event set; absolute MAE and gain versus indication, bp, intervals | Where incremental information survives advance notice |
| F3 state/quarter stability | Paired gains, intervals and counts; +1 bp state defined at origin | Whether aggregate results hide conditional failures |
| T1 data-flow table | Raw rows -> usable events -> labels -> all-horizon eligibility -> test set | Selection and reproducibility |
| T2 main comparison | Baselines, ridge, RF; MAE/RMSE, gain interval, coverage | Primary result |
| T3 controlled ablations | Feature groups, one-hour delay and common sample | Information source and timing robustness |
| T4 label-proxy sensitivity | Same predictions/events scored against official and proxy labels | Whether label substitution changes the empirical conclusion |
| Appendix diagnostic | Existing saved RF versus current and older persistence | Why the original high R² was insufficient |

Do not plot hypothetical performance curves or fill a planned table with plausible values. Current diagnostic figure: [baseline comparison](evidence/baseline_comparison.png).

## Reproduce measurements available now

Run from the repository root, using the existing environment:

```bash
v2/.venv/bin/python v2/research/reproduce_evidence.py
v2/.venv/bin/python v2/paper_plan/evidence/check_feasibility.py
v2/.venv/bin/python -m pytest -q -c v2/pyproject.toml v2/tests
```

The first script refreshes the earlier `v2/research/` outputs; copied snapshots in this package are intentionally fixed and must be explicitly refreshed if inputs change. The feasibility script reads the local source ZIP samples and writes only its JSON output. Source samples were obtained from the public URLs recorded in `archive_probe.json`; each has its official checksum beside it. No API key is needed for those files.

## Final study artifact contract

Save per-forecast `event_id`, venue/symbol, settlement time, origin, source availability and row ID, age, horizon, label availability, target, indication, prediction, model, fold, training cutoff, feature version and units. Store exclusions, fit failures, missingness and all declared comparisons. Every table must be generated from this ledger, not manually transcribed notebook output.

For every run save code/config/source hashes, environment versions, seeds, execution timestamps, source URLs/checksums and precise historical-rule assumptions. Separate acquisition from deterministic offline evaluation. Keep a protocol revision log; a hash recorded today does not make previously inspected test data untouched.

Redistribution rights for the original provider-derived ticker file need confirmation before a public release. If unavailable, release acquisition instructions, permitted derived outputs, hashes and code. An openly downloadable result table is not by itself an independently reproducible dataset.

All original inputs remain read-only. Current evidence and execution metadata are indexed in [evidence/README.md](evidence/README.md).

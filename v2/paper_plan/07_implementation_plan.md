# Implementation plan

## Deliverable boundary

Build an additive, executable study under `v2/`. This package is the implementation specification; the following modules and commands are **planned**, not currently implemented. Leave original notebooks, utilities, models, data and saved results unchanged. Do not implement the erroneous prohibitions against using the observable current indication or time-to-event features from the older build prompt.

## Proposed modules and contracts

| Proposed path | Responsibility | Interface / acceptance |
|---|---|---|
| `v2/src/funding_labels.py` | Cached official archive ingestion | `load_labels(paths) -> labels, manifest, exclusions`; checksum and units required |
| `v2/src/timebase.py` | Preserve event, arrival and source IDs | `normalize_ticker(raw) -> frame`; no synthetic market clock |
| `v2/src/settlement.py` | Origin construction and as-of event matching | `build_origin_panel(ticker, labels, spec) -> panel, exclusions`; explicit label-match offsets |
| `v2/src/features.py` | Causal feature groups and availability lineage | `make_features(panel, history, spec) -> X, lineage`; fit transforms only on training |
| `v2/src/baselines.py` | Current-indication and settled-history forecasts | All baselines use the same origin and admitted target |
| `v2/src/forecast_models.py` | Ridge and RF residual fits | Separate horizon models; fitted preprocessing included in model bundle |
| `v2/src/splits.py` | Expanding monthly cutoffs | Fold manifest with train-label cutoff, validation membership and excluded boundaries |
| `v2/src/metrics.py` | bp losses, paired gain, block intervals | Explicit rate units and reference model; stable zero-error behavior |
| `v2/src/validate.py` | Runtime data/time/lineage invariants | Reject future availability, unknown labels, conflicting event keys and nonfinite outputs |
| `v2/experiments/run_study.py` | Deterministic orchestration | One config -> versioned outputs, manifest, logs and nonzero failure exit |
| `v2/configs/settlement_study.json` | Protocol specification | Dates, horizons, staleness, publication assumptions, grids, seeds and comparison ledger |

Keep audit-only legacy transformations out of the new feature pipeline. Avoid notebook execution order as a dependency. Existing `config.py` may supply safe paths, but legacy fee constants and HAC settings do not determine the scientific protocol.

## Implementation order

### A. Freeze inputs and definitions

Record hashes of original inputs and the old audit. Acquire/check the complete label archive; store immutable downloads under `v2/data/raw/funding_labels/`. Establish label/event reconciliation and exact end date. Write E0 tests before training. Completion means a reconciliation table accounts for every candidate event, including failures.

### B. Build origins and causal baselines

Implement the event panel first. Select rows backward in arrival time within the same event, with explicit staleness. Retain actual ages. Add current indication and last-known-settlement forecasts. Verify small hand-calculated examples before processing the full history. Completion means origin counts can be explained from an exclusion ledger and baseline losses can be reconstructed row by row.

### C. Add residual features and temporal splits

Implement causal histories and availability metadata. Fit scaling/imputation only on training partitions. Use future-perturbation and boundary tests, including multiple horizons of the same event. Freeze model grids and feature groups using 2022; write the protocol hash before reading new test scores.

### D. Run E1/E2

Fit ridge/RF at monthly cutoffs and save every prediction before aggregation. Produce baseline and model tables on common events, then paired intervals and horizon curves. Record failed fits; do not silently replace them with the baseline. A failed fit is either fixed and rerun under a recorded implementation revision or reported with failure/coverage counts.

### E. Run E3/E4 and conclude

Reuse the forecast ledger for state and quarter analysis. Refit only controlled feature/delay variants. Report all declared comparisons. Decide between a positive result, a bounded negative result and inconclusive evidence according to effect estimates, not preferred narrative.

Also run E3b by rescoring the identical saved predictions against terminal-indication proxies on a matched sample. Save this as a separate diagnostic table. Do not train on the proxy or mix its scores into the official-label leaderboard.

### F. Package the paper

Generate tables and figures from the result ledger, replace every pending result slot, reconcile citations and release a minimal rerun command. Keep an appendix for the original baseline reversal. A paper is ready for technical review only after another clean-environment run reproduces its tables.

## Planned command contract

The following CLI is a design target, **not a command that works today**:

```bash
v2/.venv/bin/python -m v2.experiments.run_study --config v2/configs/settlement_study.json --stage validate-data
v2/.venv/bin/python -m v2.experiments.run_study --config v2/configs/settlement_study.json --stage baselines
v2/.venv/bin/python -m v2.experiments.run_study --config v2/configs/settlement_study.json --stage models
v2/.venv/bin/python -m v2.experiments.run_study --config v2/configs/settlement_study.json --stage report
```

Use `v2` consistently as an import namespace or use the existing `src` convention with a documented working directory; resolve this once during scaffolding. Do not mix import conventions incidentally across notebooks and scripts.

## Scope controls

No LSTM/Transformer search, new asset, paid feed, live trading or full execution simulator is necessary for the minimum study. If the closest paper already contains the proposed evaluation, add a justified replication/robustness question before expanding model count. If provenance is not recoverable, narrow the study to indication updates or acquire the minimum missing data; do not certify a settlement benchmark from proxies.

# Revised settlement study artifacts

Protocol `2026-10-01-v2.1`; execution code commit `00a8a9c`. Start with
[REPORT.md](REPORT.md) and the [execution record](../../paper_plan/16_hardening_execution.md).
The September 30 run remains unchanged in the adjacent `settlement_study/` folder.

Each `_stages/<stage>/receipt.json` records the execution identity and exact hashes
of its dependencies and new outputs. A stage directory is published atomically;
its top-level artifact links are conveniences. Rerunning a completed stage
verifies hashes without retraining or rewriting. All generated scientific
artifacts are covered by stage receipts; this README is explanatory documentation.

| Artifact | Purpose |
|---|---|
| `manifest.json` | Official source URLs, archive/checksum digests, code/config hashes and environment |
| `coverage.json`, `*_exclusions.csv` | Data admission, warm-up and validation/evaluation boundary exclusions |
| `baseline_validation_candidates.json` | Every EWMA candidate on the same monthly validation cohort as models |
| `validation_candidates.json`, `selection.json` | All candidate scores and frozen selected parameters |
| `baseline_folds.csv`, `folds.csv` | Monthly training maturity, sample counts and eligible origins |
| `panel.csv.gz`, `predictions.csv.gz` | Local row-level lineage, features and all fixed forecasts |
| `publication_delay_sensitivity.json` | Exact equality checks for 0/5/15-minute label-availability assumptions |
| `forecast_replay.json` | All primary four-hour ridge folds and the first RF fold replayed from saved inputs |
| `forecast_artifact_manifest.json` | Reporting dependencies and complete execution identity |
| `scores.csv`, `paired_contrasts.csv` | MAE/RMSE/bias/skill, pointwise calendar-block intervals and Holm tests |
| `proxy_label_scores.csv`, `state_quarter_scores.csv`, `staleness_scores.csv` | Label-proxy and descriptive conditional evidence |
| `lead_time.png` | Measured lead-time error and incremental-gain figure |
| `reproduction_check.json` | Comparison against a separate full pipeline execution from cached sources |

The reproduction uses the same machine and package versions, not independently
collected market data. Raw public archives and provider-derived `.csv.gz` files
remain ignored pending redistribution review. The alternate reproduction config
writes to an empty ignored directory in a fresh checkout. See the [run commands](../../README.md)
before attempting to execute into a directory containing committed evidence.

The first hardening attempt (`v2`) was interrupted during selection after finding
a verification tolerance mismatch. Its data/baseline stages remain local; it
never committed model forecasts or a report. This `v2.1` run is a fresh execution.

# Results ledger

> Historical design/preparation record. Current execution status, corrections and
> evidence are maintained in [the hardening record](16_hardening_execution.md)
> and [the revised report](../results/settlement_study_v2_1/REPORT.md).

**30 September 2026 execution update:** the original pending ledger below is a historical planning record. Current E0–E4 results and remaining limitations are in [the generated report](../results/settlement_study/REPORT.md), with machine-readable scores and source manifests beside it.

Status codes: **VERIFIED** = measured from available inputs; **REPORTED** = an external/legacy claim not independently reproduced; **PENDING** = not run. Never copy pending entries into an abstract as results.

## Verified local evidence

| ID | Result | Scope / artifact |
|---|---|---|
| R01 | 42,179 rows; 5,256 event IDs | Original dataset; [audit measurements](evidence/evidence.json) |
| R02 | RF R² 0.974973; current persistence R² 0.980076 | Same 7,797 saved next-row predictions |
| R03 | RF MAE 0.140612 bp; current persistence 0.094347 bp | [Score table](evidence/saved_prediction_scores.csv) |
| R04 | RF MSE 25.6158% higher; MAE 49.0364% higher | Descriptive artifact comparison; no inferential claim |
| R05 | 7,793 rows align current reconstructed value and next raw target | Raw current/next both nonmissing; four saved rows excluded from this alignment check |
| R06 | 3,664/5,256 event groups have changing indications | 69.71%; not merely repeated final payments |
| R07 | 8,531 backward-filled cells and 1,118 funding values zeroed | Legacy preprocessing; effects on clean forecast performance unmeasured |
| R08 | Synthetic clock ends about 130 hours ahead of arrival time | Clock must be repaired before event-level modeling |
| R09 | SARIMAX R² -4.807223 | Saved output; MAE 1.7871 bp conditional on documented 10^6 scaling |
| R10 | All 5,256 IDs have four eligible horizons with age ≤65 min | [Feasibility output](evidence/feasibility.json); not label validation |
| R11 | 3 public ZIPs verified; 276 labels; 275 matched indications | Purposively selected months, not a generalization test |
| R12 | 47/275 sampled terminal indications differ by >0.01 bp | Full sample comparison in the data-contract document |
| R13 | 14 existing tests passed | Path/configuration suite, not modeling tests |

## New model experiments

| ID | Status | Primary estimate | Interval / test | Required artifact |
|---|---|---|---|---|
| E0 full-period panel | PENDING; sample probe only | Not measured | Not applicable | Complete manifest, mapping and exclusions |
| E1 four-hour ridge/RF | PENDING | Not measured | Not run | Per-event predictions + main table |
| E2 matched lead-time curve | PENDING | Not measured | Not run | Same-event horizon comparisons |
| E3 feature/delay ablation | PENDING | Not measured | Not run | Paired ablation outputs |
| E3b label-proxy sensitivity | PENDING | Not measured | Not run | Identical frozen predictions rescored on matched labels |
| E4 observable-state/time | PENDING | Not measured | Not run | Counts, conditional gains and uncertainty |
| E5 robustness | PENDING | Not measured | Not run | Tail/window/seed comparison |
| E6 fresh-time / DAR replication | PENDING | Not measured | Not run | Locked replication outputs |

## Row template for completed experiments

Record experiment ID, code commit/hash, protocol hash, source manifest hash, execution timestamp, target/units, model/hyperparameters, fold dates, sample exclusions, baseline, effect estimate, uncertainty method, interval/p-value, multiplicity family, errors and runtime. Link the raw predictions and generated table. State whether the analysis was planned, revised before test access or exploratory after results.

Do not replace the existing RF score with a future score from a different target. Retain both with separate task labels: next-update diagnostic versus settled-funding forecast.

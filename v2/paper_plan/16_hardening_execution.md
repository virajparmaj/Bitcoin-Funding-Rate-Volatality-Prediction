# Settlement study hardening execution record

Protocol: `2026-10-01-v2.1`. Execution code frozen in commit `00a8a9c` before model
selection and test forecasts. Both complete runs passed all stages, including
exact publication-delay checks, saved-forecast replay and report generation.
The independent full reproduction passed all 19 artifact comparisons.

## Requirement-to-evidence map

| Requirement | Evidence |
|---|---|
| Preserve original evidence | Original-run SHA-256 check and Git protected-path diff |
| Official archives, checksums and lineage | Revised manifest, coverage and origin exclusions |
| Identical monthly validation eligibility | EWMA candidate ledger and adversarial boundary test |
| Frozen selection and matured monthly folds | Selection identity, folds and stage receipts |
| E1/E2/E3/E3b/E4 saved predictions | Forecast ledger, score/contrast/proxy/state tables |
| Paired uncertainty and multiplicity | Scores, declared comparison family and generated report |
| Exact publication-delay check | Sensitivity receipt, 0/5/15-minute checks, boundary exclusions |
| Deterministic replay and full reproduction | Forecast replay and reproduction comparison records |
| Immutable outputs and safe interruptions | Stage integrity tests and per-stage receipts |
| Versioned execution and current status | Revised/reproduction configs, README and decision log |

## Remaining scientific limits

This is retrospective BTCUSDT research on previously inspected years. Original
hourly field aggregation lacks verified message-level provenance; historical
publication delay is assumed. Fresh-time/external replication, closest-paper
full-text verification, DAR replication and full seed/window/tail robustness
remain unexecuted. Pointwise intervals do not establish simultaneous equivalence;
forecast error reductions do not establish profit or algorithmic novelty.

## Executed evidence

- **53 tests passed**; Black and Ruff checks passed for changed execution/test
  modules. The demo production build and Git whitespace checks passed.
- **58 archives**, **5,298 official labels**, **5,248 admitted events** and
  **20,992 origins** before monthly evaluation exclusions.
- EWMA and every model candidate use **1,083 validation events per horizon**.
  EWMA span 3 remains selected; all selected parameters match the original run.
- **248,752 predictions**, **352 model fold records**, **1,947 forecast events**,
  and **1,889 common events** across all horizons and availability variants.
- Exact 0/5/15-minute delay checks preserve features and validation/evaluation
  event-horizon membership. Boundary exclusions: **12 validation events / 48
  origins**, **22 evaluation events / 88 origins**.
- Each run passed **23 saved-panel model/fold replays**. Separate full runs match
  all 19 compared artifacts, including every forecast, eligibility record,
  validation ledger and numerical table (`rtol=1e-9`, `atol=1e-12`).
- Re-executing every completed stage preserved both SHA-256 hashes and file
  modification times. All **57 original evidence files** remained byte-for-byte
  unchanged. The largest native-rate forecast difference from the September run
  is **2.1684e-19**, below the declared numerical tolerance.
- The generated lead-time figure was visually inspected. Its intervals are
  labeled pointwise and use consistent model colors.

## Measured conclusion

On 1,889 common events, four-hour MAE is **0.246866 bp** for the current indication,
**0.238253 bp** for ridge and **0.232464 bp** for RF. The primary ridge gain is
**0.008613 bp**, with a 95% seven-day calendar-block interval of
**[-0.007970, 0.028633] bp**. Positive primary incremental skill is not established.
The one-hour RF secondary gain is **0.019581 bp**, Holm-adjusted **p=0.013797**;
it does not replace the primary comparison or imply economically useful profit.

The validation correction changes EWMA's selection cohort, not its selected span.
The revised study retains the original rounded results and conclusions. The full
reproduction uses the same machine/environment and cached source data, so it
establishes computational reproducibility rather than external validity.

See [the generated report](../results/settlement_study_v2_1/REPORT.md),
[reproduction checks](../results/settlement_study_v2_1/reproduction_check.json),
and [local validation evidence](evidence/hardening_validation.json).

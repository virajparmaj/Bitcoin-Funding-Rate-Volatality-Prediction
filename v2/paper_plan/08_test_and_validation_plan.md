# Tests, validation and current execution status

> Historical design/preparation record. Current execution status, corrections and
> evidence are maintained in [the hardening record](16_hardening_execution.md)
> and [the revised report](../results/settlement_study_v2_1/REPORT.md).

## Executed during preparation

| Check | Result | What it actually establishes |
|---|---|---|
| Existing v2 test suite | **14 passed** | Path guards, input presence and configuration invariants |
| Prior evidence script | Completed in the prior assessment; input hashes verified | Saved-prediction arithmetic and target alignment |
| Three official archive downloads | **3/3 checksum matches**, 276 rows | Integrity of these source samples, not complete historical coverage |
| Offline feasibility script | Completed with assertions passing | Causal row selection under recorded-arrival assumption, sample label reconciliation, unchanged raw input hash |

The 14 passing tests are not a forecast-validity certificate. No new residual model was trained and no paired significance/equivalence test was run. Documentation validation is recorded separately in the evidence folder.

Existing tests were invoked from the repository root with:

```bash
v2/.venv/bin/python -m pytest -q -c v2/pyproject.toml v2/tests
```

## Scientific validation tests to implement

All rows below are **planned / not run**. Use tiny hand-computed fixtures and adversarial counterexamples where possible; do not only assert that a function exists or that a source string is absent.

| ID | Scenario | Required behavior |
|---|---|---|
| T01 | Known native rate 0.0001 | Equals 1 bp; CSV values and losses use documented units |
| T02 | Source row one microsecond after origin | Excluded, even when nearer than the prior row |
| T03 | Row exactly at origin | Included under declared `<=` rule |
| T04 | Latest row belongs to next event | Never borrowed for the current event |
| T05 | Source age 65 minutes and just over | Boundary behaves exactly as configured; exclusion is recorded |
| T06 | Missing rate or label | No fabricated zero/forward-filled outcome |
| T07 | Conflicting duplicate label ID | Quarantined or hard failure, never arbitrary `last()` |
| T08 | Archive calc time differs by milliseconds | Exact raw time preserved; unique documented candidate mapping accepted |
| T09 | Ambiguous or excessive label offset | Rejected with a reason and source trace |
| T10 | Modify all observations after a chosen origin | Features at that origin remain identical |
| T11 | Modify validation/test distribution | Earlier fitted scaler/imputer parameters remain identical |
| T12 | Label matures after training cutoff | Excluded from fitting and settled-history features |
| T13 | Same event has four horizons | All stay in the same split; bootstrap samples them together |
| T14 | Causal history crosses a data gap | No unbounded fill or invented hourly observation |
| T15 | Reconstruct current indication from lags/MA | Allowed if observable; an injected future target must fail |
| T16 | Constant feature or +1 bp predictor | Logged as a diagnostic; not automatically called leakage |
| T17 | Model equals indication | MAE gain and skill are zero, except undefined zero-denominator skill handled explicitly |
| T18 | Known toy paired losses | Metrics match hand calculation; reversing model/reference reverses gain sign |
| T19 | Calendar gaps in bootstrap | Blocks retain calendar spacing and model/event pairing |
| T20 | Null synthetic forecasts / repeated candidates | Inference and multiplicity checks do not systematically manufacture improvement; report Monte Carlo uncertainty |
| T21 | One-hour availability delay | Recompute origins/features and common eligibility, not merely relabel timestamps |
| T22 | Wrong checksum or malformed source | Abort before model fitting; original file stays unchanged |
| T23 | Deterministic rerun with fixed seed/config | Identical eligibility, folds and predictions within documented numeric tolerances |
| T24 | Failed estimator fit | Failure logged and visible; no silent baseline substitution |
| T25 | Proxy-label sensitivity | Identical predictions/events under both labels; final indication never available as a forecast feature |

## Study acceptance

E0 passes when every included event has a unique auditable label and every feature has a defensible availability bound. E1–E4 pass when all declared outputs are generated from the same saved predictions, eligibility changes are explained and uncertainty reflects temporal dependence. Passing means the study is correctly executed, not that the models win.

Scientific results must retain true extremes and unsuccessful comparisons. A result that fails the practical-gain criterion is valid evidence if measured precisely; broad intervals remain inconclusive.

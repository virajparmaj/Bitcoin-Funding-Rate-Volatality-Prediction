# Decision log and changes from the earlier assessment

Date: 25 September 2026. This log records planning decisions, not completed forecast experiments.

| Decision | Evidence / reason | Consequence |
|---|---|---|
| Keep one contract and simple residual models | Existing data/model families suffice for a focused comparison | Avoid unnecessary new architectures and feeds |
| Current indication is the primary baseline | Verified target shift and Tardis field semantics | Replace the older-lag headline comparison |
| Obtain actual labels before training | Sample terminal indications differ from archived settlements | Final ticker rows cannot serve as exact outcomes |
| Public archive route is feasible | Three month ZIP/checksum pairs verified | No initial paid ticker repull is required merely to obtain labels |
| Preserve scheduled event and archive calc time separately | Sample offsets reach 25 ms | Exact-time equality is insufficient; matching must be explicit and audited |
| Include the final 18 October 2024 event | Features end just before its settlement | Earlier “through 17 October” wording needs an explicit event cutoff |
| Provisional max age 65 minutes; 30/60 sensitivities | All 5,256 IDs feasible; stricter coverage differs | Report effective lead time and coverage, not nominal horizon alone |
| Ridge at four hours is the primary comparison | Prevent test-driven model/horizon selection | RF and other horizons are declared secondary tests |
| Include a low-cost state/time analysis | Strengthens empirical specificity with existing predictions | Required when claiming state dependence |
| Add label-proxy rescoring, E3b | Sample labels differ and saved predictions can be reused | Measure whether the distinction changes errors/rankings without training extra models |
| Do not frame auditing or lead-time comparison as an invention | New audit preprint and practitioner guide overlap | Claim a measured residual-information result only |
| Treat evaluation as retrospective | Original sample already inspected | Later untouched replication is recommended |
| Full model implementation deferred to planned work | User requested research strengthening and an implementation plan | Only diagnostics, feasibility probes and documentation executed here |

## Older build-prompt interpretations that must not govern new implementation

`V2_BUILD_PROMPT.md` and the original v2 README preserve earlier interpretations that are now contradicted or unsupported: reconstructing current q is not automatically future-target leakage; null following-event estimates do not remove the upcoming indication; dropping days is not established as the clock-drift explanation; ties need not vanish at settlement; constants and time-to-event features are not automatically invalid; HAR-RV is not a settled-level baseline without a variance target; eight settlement lags are not one day; fee/MAE ratios are not profitability tests.

The original files remain unchanged as historical context. This package proposes corrected scientific requirements for the future additive build. Its public-source feasibility requests are read-only research checks, not the older prompt's proposed paid/cross-exchange acquisition phase.

## Outstanding uncertainties

Closest Inan full text; full-period official label coverage; historical funding-rule changes; raw-message provenance of hourly aggregation; exact historical label publication delay; redistribution rights; clean residual forecast performance and its precision. No date, readiness promise or model score should be invented to close these gaps.

## Implementation revision — 30 September 2026

- December 2019 archive returned HTTP 404. Start label acquisition in January 2020 and exclude incomplete nine-settlement warm-up explicitly.
- Full-period calc-time offsets are at most 47 ms. Use a one-second bounded mapping, preserve original timestamps and reject duplicates/schedule deviations.
- Exclude month-boundary events whose origins precede the monthly fit; otherwise the fitted model would use information from after its claimed decision.
- Feature ablations reuse the full-model validation selection to isolate information removal at fixed hyperparameters. This differs from independently tuning each reduced feature set.
- Primary analysis uses the common intersection across all declared availability variants and horizons. Report both total forecast coverage and common-cohort counts.
- Subgroup/quarter analysis is descriptive with pointwise intervals. No simultaneous equivalence, universal unpredictability or economic-value conclusion is made.
- Source code, source hashes, tuning choices and aggregate outputs are committed; raw provider-derived panels/predictions remain local pending redistribution review.
- The executable config and generated report supersede historical pending statuses. Later-time/DAR replication and full seed/window/tail robustness remain separate work. The 0/5/15-minute publication-delay check was executed and produced identical histories and fold membership.

## Reproducibility hardening — 1 October 2026

Protocol `2026-10-01-v2` implements the approved completion pass. The original
`2026-09-30-v1` config and result directory remain immutable historical evidence.
Before any revised test scoring, the following corrections were fixed:

- EWMA selection now uses the same monthly 2022 eligibility as ridge/RF/history
  ridge. Whole events are partitioned before horizon filtering. Every EWMA span
  enters the validation ledger; ties follow configured grid order.
- Every execution module, package version, config, ticker, archive and checksum
  is part of the execution identity. Completed stages verify without rewriting;
  changed identities require another output directory.
- Stages commit immutable directories atomically. Top-level artifact links point
  to those directories. Failed computations cannot replace earlier evidence;
  a committed stage can repair links after an interrupted publication.
- Delay diagnostics compare exact keyed features and both validation/evaluation
  membership. Approximate equality no longer establishes invariant predictions.
- Both verification CLIs accept `--config`. A separate full reproduction uses
  identical scientific settings and fresh selection/fitting from cached sources.
- Scientific periods, grids, seed, losses and inference family are unchanged.
  This is a retrospective protocol correction, not an untouched holdout.
- The revised date reflects actual implementation on 1 October. The plan's
  proposed September 30 revision date was not used to backdate execution.

Results and completion evidence will be appended after execution. No model or
hyperparameter changes will be made in response to revised test performance.

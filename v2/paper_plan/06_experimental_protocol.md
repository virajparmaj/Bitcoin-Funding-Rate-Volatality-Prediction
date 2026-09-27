# Experimental protocol to implement

**Status: specified, not executed.** E0 has only a three-month feasibility probe. This document refines the earlier broad experiment menu into a locked minimum study.

## Fixed choices

- Primary model/comparison: residual ridge versus current indication at four hours; positive MAE reduction means improvement.
- Secondary model: residual RF. Secondary horizons: one, two and six hours. Use separate horizon fits.
- Training: 2020–2021; tuning/selection: 2022; retrospective evaluation: 2023 through the last verified event on 18 October 2024. Fit at each month start using only labels available before that cutoff. First test fit may include matured 2022 labels.
- Training history expands; features update causally between fits. Keep all horizons of an event in the same temporal partition; purge events whose labels are not yet available. Record one-event boundary gaps rather than imposing unexplained random embargoes.
- Freeze features, grids, seeds and comparisons after validation. These years have been inspected before: call this retrospective evaluation, never an untouched holdout.
- Eligible feature observations must precede the origin, belong to the same upcoming event and be no more than 65 minutes old. Report 30/60-minute sensitivities. Main horizon curves use identical event IDs.

## Models and compact tuning budget

| Method | Inputs / configuration |
|---|---|
| Current indication | Forecast `q_s,h`; residual prediction zero |
| Last settled funding | Latest verified label available at origin |
| Settled EWMA | Spans 3, 9, 21 events, selected on 2022 MAE |
| Median / +1 bp references | Training median; +1 bp only with historically applicable mechanics |
| History ridge | Settled lags 1, 2, 3, 9; alpha 0.1, 1, 10, 100 |
| Residual ridge | Standardized features; alpha 0.1, 1, 10, 100; primary mean model |
| Residual RF | 300 trees; depth 3, 6, unlimited; min leaf 5 or 20; max_features 1.0; seed 42 |

These are proposed grids, not tuned results. Use exactly the same folds and eligibility when selecting hyperparameters. Ridge/RF train with squared error but are selected and primarily evaluated by MAE; disclose this mismatch. Do not search more models when test results disappoint. An optional robust-loss fit belongs in sensitivity analysis, not silent replacement of the main method.

Feature groups: current indication and age/event hour; observed indication differences/variability within the event; previously matured settlements and volatility; mark/index spread, price returns and OI changes. Funding differences use native or bp units, not percentage changes across zero. Mark/index spread is a proxy, not the exchange's impact-price premium index. Features requiring unverified fields must have an omission/delay sensitivity.

## Statistics

For each matched event store paired absolute and squared losses. Primary effect is mean MAE reduction in bp with a 95% moving-calendar-block bootstrap interval, 5,000 draws, seed 42, seven-day blocks. Preserve all horizons/models from a sampled event together; gaps stay in the calendar and are not compressed into fictitious adjacent events. Check 3/14-day block sensitivity. With insufficient blocks in a state, report insufficient precision.

Report RMSE, squared-error skill relative to the indication, signed bias, counts and coverage as secondary outcomes. Avoid MAPE and unstable skill ratios when baseline loss is near zero. DM/HAC is a secondary check; lag choices must respect settlement time units and dependence.

Declare a comparison ledger before test evaluation. Apply Holm correction across secondary ridge/RF horizon comparisons and declared state/ablation tests; accompany them with effect sizes and intervals. For family-wide negative claims use simultaneous bounds, not separate unadjusted intervals. Treat state/quarter findings as descriptive if they were not frozen. Seeds/folds are not independent market replicates.

## Minimum experiments and acceptance

| ID | Change | Output | Required evidence / failure condition |
|---|---|---|---|
| E0 | Verify labels, field meanings and availability | Coverage/provenance table, exclusion ledger, origin timeline | Every admitted row has unique label and causal lineage; unresolved ambiguity narrows or blocks settlement claims |
| E1 | Compare baselines, ridge and RF at four hours | Main table, paired intervals, cumulative loss difference | Same events and units; wide intervals are inconclusive |
| E2 | Vary lead time 1/2/4/6 hours | Matched-event error and incremental-skill curves | A difference disappearing on common coverage cannot support a horizon claim |
| E3 | Add feature groups; delay availability one hour | Ablation table on common eligible events | Gains requiring future values or changing outcomes must be withdrawn |
| E4 | Slice fixed forecasts by origin-known state and calendar quarter | State/quarter table with counts and uncertainty | Too few transitions or one dominant episode weakens generality |

E0–E3 are essential. E4 is strongly recommended at low marginal cost and becomes essential if the paper's claim mentions market-state dependence. Define the flat state as observed rate equal to +0.0001 within source precision; use an absolute tolerance of 1e-10 native rate in the initial implementation and report its sensitivity. Define high trailing volatility using training-derived thresholds only. March 2024 stays in aggregate results and is a transparently selected descriptive stress slice.

### E3b: label-proxy sensitivity with no additional training

For the same already frozen baseline/ridge/RF predictions, compare scoring against the official settlement label with scoring against the final observed pre-settlement indication. Use identical event IDs and retain the proxy's exact observation time and age. Report changes in MAE gain, RMSE and model ordering at each horizon, with paired calendar-block intervals for the difference in measured gain. Include these declared comparisons in the secondary multiplicity ledger. Official labels remain the main outcome; the proxy is an explicitly retrospective diagnostic and must never enter forecast features.

This supports a narrow claim about whether label substitution changes the empirical conclusion in this dataset. If differences are negligible or rankings do not change, do not claim a consequential ranking reversal. The existing three-month discrepancy check motivates the test but does not establish its result. Because this only requires rescoring saved predictions, it is a useful inexpensive addition to the minimum study.

## Additional experiment specifications

The full specifications below retain all required change/model/split/metric/inference/output/falsification details. Where the earlier common protocol differs, the explicit primary-model, cutoff and label-availability rules above take precedence.

### Detailed experiment cards

### Common protocol

Use settled rate `F_s` at actual event time `T_s`. For lead time `h`, define origin `o_(s,h) = T_s − h`. Select only source observations with availability time **at or before the origin** and the correct upcoming event ID. Let `q_(s,h)` be the last eligible exchange indication. Target the residual `u_(s,h) = F_s − q_(s,h)`; the final model prediction is `q_(s,h) + predicted_u_(s,h)`.

Use h = **4 hours as primary**; h = **1, 2, 6 hours as secondary**. Six hours is more defensible than demanding eight-hour origins from this particular reduced file, whose first observations for many events arrive after the previous settlement. Add eight hours only when event-valid observations genuinely exist. Never select the closest observation if it falls after the origin. Record age and coverage; reject rows older than a preregistered tolerance, initially 65 minutes, with a stricter sensitivity run.

Development: 2020–2021 initial training; 2022 validation for feature/hyperparameter choices. Lock the protocol before evaluating 2023 through 17 October 2024 with monthly expanding-window refits. Within each month, parameters remain fixed, but features and baselines use observations available at each origin. Training labels must be available strictly before the fit cutoff. All rows for one settlement stay in the same evaluation partition. At fold boundaries purge overlapping label intervals; a conservative one-settlement gap is acceptable but must be reported. Earlier observed market history can still construct causal test features.

**These years have already been inspected in v1 and v2.** Call this retrospective locked-protocol evaluation, not an untouched confirmatory holdout. A later untouched sample materially strengthens the paper. March 2024 can be a transparently selected descriptive stress slice, not a freshly discovered independent stress test. Keep it in aggregate performance rather than removing difficult events from the headline score.

Core baselines: current indication; last confirmed settled rate; validation-tuned EWMA of confirmed settled rates; training-median settled rate; and a +1 bp mechanism reference where historically applicable. Include a low-order AR/ridge history model. Train two residual models: regularized linear regression and RF, using existing feature families after availability repairs. The small feature set comprises current indication, observed within-event changes, past settled rates, past indication variability, mark/index spread, mark-price returns, OI changes, age/missingness and event hour. The mark/index spread is only a proxy, not the exchange's impact-price premium index.

Use squared-error skill `1 − sum(model_error²)/sum(baseline_error²)` and MAE reduction in bp. Primary loss is absolute error; RMSE and squared-error skill are secondary. Match training objectives to losses or report the mismatch explicitly. RF squared-error fits are not automatically MAE-optimal. Do not use MAPE near zero. Report signed bias, sample counts and coverage. For nearly zero baseline error, give absolute error rather than unstable skill ratios.

For inference, use paired loss differences indexed by settlement. Resample contiguous **seven-day blocks (21 settlements on the standard schedule)**, retaining all horizons/models of an event together; use 5,000 bootstrap draws with a recorded seed. Report 95% intervals and sensitivities to 3- and 14-day blocks. This supports uncertainty conditional on the evaluated forecasting procedures; it does not make a retrospectively chosen design confirmatory. A [Diebold–Mariano](https://doi.org/10.1080/07350015.1995.10524599) comparison with HAC errors is a secondary check, with lag choice tied to observed dependence and time units rather than the existing constant. Apply Holm correction to the declared secondary model/horizon comparisons. Do not treat folds or random seeds as independent market samples.

For a negative conclusion, predeclare a practical MAE-gain threshold; **0.05 bp** can be used as an explicit research resolution, with **0.01 and 0.10 bp** sensitivity, but must not be called a fee-derived profit threshold. An upper confidence bound below the chosen threshold supports “no improvement as large as this threshold,” not universal unpredictability. A nonsignificant p-value alone is inconclusive.

### ESSENTIAL

**E0 — Verify field meaning, timestamps and realized labels.**

- Independent change: compare the last observed indication with authoritative settlement records; compare original synthetic time, recorded arrival time and conservative availability delays.
- Models: none; this is a measurement gate.
- Sampling: all eligible events, plus manual traces from early/late years, extremes, missing values and event boundaries. Obtain historical settled BTCUSDT rates for the existing date span from the public exchange history endpoint/archive; this is a small label supplement, not a replacement feature dataset.
- Metrics/tests: matched/missing event counts, duplicate IDs, last-observation age, mismatches in bp, boundary violations, rule/schedule changes; deterministic assertions rather than a significance test.
- Output: data-flow/availability diagram and coverage/proxy-discrepancy table.
- Claim supported: the benchmark forecasts actual payments from information available at the origin.
- Invalidating outcome: substantial unrecoverable timing/label ambiguity. If unresolved, explicitly restrict the paper to future indicative updates; do not label terminal proxies as realized funding.

**E1 — Establish clean benchmark skill at the primary four-hour origin.**

- Independent change: baseline versus residual ridge versus residual RF; add the history-only AR/ridge comparator.
- Split: common protocol above, same eligible test events for every method, no outcome imputation.
- Metrics/tests: primary MAE gain in bp; RMSE, bias and skill; paired block intervals; declared model-comparison multiplicity.
- Output: principal model/baseline table with counts and uncertainty; monthly cumulative loss-difference plot.
- Claim supported: a measured presence, absence within a tolerance, or uncertainty about incremental predictive information at four hours.
- Weakening outcome: confidence intervals too wide to distinguish improvement, non-improvement or practical equivalence; performance that depends on one short episode.

**E2 — Measure the lead-time profile on the same settlements.**

- Independent change: h = 1, 2, 4, 6 hours; fit distinct horizon models with the same declared family/selection protocol.
- Models: E1 set. No new architectures.
- Split: same temporal folds; primary horizon comparisons on the intersection of event IDs with valid observations at all horizons. Also report each horizon's maximum-coverage sample to quantify selection.
- Metrics/tests: baseline and model MAE/RMSE by h; incremental gain by h; paired bootstrap differences between lead times and multiplicity correction for secondary comparisons.
- Output: two-panel lead-time curve: absolute error and gain over the current indication, with confidence bands and counts.
- Claim supported: where advance notice and incremental information coexist in this dataset.
- Weakening outcome: apparent curves disappear on matched coverage or after actual availability is enforced; a constant negligible result may still support a narrowly bounded negative finding.

**E3 — Run a controlled information-set and availability ablation.**

- Independent change: current indication only; plus indication history/settled history; plus price/OI features. Separately compare source-based availability with a one-hour conservative delay. Refit each feature-group model.
- Models: residual ridge and RF; the no-change residual baseline remains zero.
- Split: identical event IDs/folds; a common missingness mask for feature ablations. A secondary deployment-like run may allow different coverage, clearly labeled.
- Metrics/tests: paired MAE changes, bootstrap intervals and Holm adjustment for declared group tests. Include all true test extremes.
- Output: feature-group and availability table. Legacy scores appear separately as diagnostics, not on the same clean leaderboard.
- Claim supported: whether any improvement comes from incremental market variables and survives admissible timing.
- Weakening outcome: gains require unavailable observations, target-derived features, or changed evaluation coverage. Then withdraw the predictive claim.

The minimum empirical paper is E0–E3 plus a reproducible release and a resolved full-text novelty comparison. This is a **plausible minimum study**, not a guarantee that reviewers will consider a one-contract negative result sufficiently novel.

### STRONGLY RECOMMENDED

**E4 — State and temporal stability.** Change the evaluation stratum, not the trained model: indication at +1 bp versus elsewhere; low/high trailing funding variability defined using training thresholds; calendar quarters; descriptive March 2024 slice. Use E1/E2 forecasts, origin-observable state labels, per-stratum MAE gain/counts and block-bootstrap differences between strata. Show a state-by-lead heatmap and quarterly forest plot. Supports localization of skill/failure. Weakens the story if gains are concentrated in an unrepeatable handful of events. Realized-tail slices may be diagnostic but are not deployable regime definitions.

**E5 — Tail-preserving training and sensitivity to refitting.** Vary training only: raw targets, training-only robust loss/winsorized fit, and the historical zeroing path; always score against untouched true outcomes. Compare expanding with a fixed trailing 365-day window and RF seeds 0, 1, 2, 3, 4. Use primary-horizon ridge/RF; existing splits; overall and high-|F| MAE/RMSE, bias, paired block intervals. Report a compact robustness table. Supports resilience to preprocessing and fit choices. Weakening outcome: the central effect changes sign across reasonable specifications or is smaller than fit instability. Seed variation describes algorithm stability, not extra independent evidence.

**E6 — Untouched-time replication and closest-model comparison.** Freeze choices, obtain later BTCUSDT ticker/settlement observations using the same schema, and evaluate once. In the settlement-history comparison, reproduce a DAR specification from the closest paper once full methods are available; do not describe AR(1) as an exact DAR replication. Compare matched origins, MAE/RMSE and dependence-aware intervals; publish a replication table with protocol differences. Supports temporal transport and literature distinctness. Failure narrows the claim to the original historical sample. A public settled-only extension cannot replicate the intraperiod-estimate question without its corresponding historical indications.

### OPTIONAL

**E7 — Residual uncertainty.** Independent change: constant/rolling/EWMA variance versus AR–GARCH/GJR/EGARCH for a fixed mean forecast's sequential residuals. Train only on previously matured forecasts, use the same origins and temporal splits, and compare QLIKE on a declared squared-residual proxy plus interval coverage/width or proper predictive scores. Use block intervals and report convergence failures. Output: coverage/error-size calibration figure and variance-score table. Supports useful uncertainty modeling even if mean gains are small. Invalidating result: nonpositive forecasts, poor calibration, or no advantage over a rolling variance. HAR-RV belongs here only with a defensible realized-variance target, not as a funding-level baseline.

**E8 — Decision value.** Change only the forecast source inside one preregistered policy: model, current indication, settled persistence, always-hold carry and no-trade. Use identical event times, sizing and trade rules. Specify positions, both hedge legs, actual funding cash flows, execution prices, holding periods, turnover, borrow/collateral costs and mark-to-market/basis P&L. Evaluate net returns, drawdown, turnover and incremental P&L with dependence-aware intervals; sweep explicit fee/slippage scenarios. Output: cost-versus-incremental-return curve and P&L decomposition. Supports policy-specific usefulness. Invalidating outcome: the advantage disappears under modest friction or depends on impossible fills. Existing mark/index values alone do not prove executable profitability.

**E9 — External asset/venue validation and runtime.** Freeze the protocol and compare BTC with ETH on the same venue, then another venue only after normalizing schedules and field semantics. Reuse core models; use calendar-matched folds and separately report within-venue and transfer fits. Metrics: forecast gains with event/day-block intervals, fit time, inference time and peak memory measured on fixed hardware. Output: transport table and accuracy/runtime plot. Supports bounded generalization or an efficiency finding. Weakening outcome: effect fails outside BTCUSDT or runtime differences have no meaningful operational consequence. Energy claims require actual energy measurement.

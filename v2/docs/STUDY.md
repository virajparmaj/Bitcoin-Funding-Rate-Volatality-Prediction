# Funding settlement study

## Question and scope

Does a simple model improve on the current exchange indication of the upcoming
BTCUSDT funding settlement at 1, 2, 4 or 6 hours before the payment? The contribution
is a bounded empirical measurement, not a new algorithm, publication-priority
claim or trading-profit result. Evaluation is retrospective: these years were
inspected before protocol selection.

The [config](../configs/settlement_study_v2_1.json) defines the executable protocol;
the [generated report](../results/settlement_study_v2_1/_stages/report/REPORT.md)
contains results. See the [README](../README.md) for environment and run commands.

## Data and availability

The source ticker file contains recorded arrivals and an upcoming funding-event
identifier. For each origin, select the last same-event observation available at
or before the origin, with a maximum age of 65 minutes. Never borrow a future
observation, invent an observation clock or impute an outcome. Preserve extremes
and write exclusion reasons and source-row lineage.

Official Binance monthly funding ZIPs and checksum files supply realized labels.
Scheduled event time, actual calculation time and assumed label availability are
separate. An at-most-one-second reconciliation maps calculation jitter to the
00:00/08:00/16:00 UTC schedule; ambiguous matches, duplicates or schedule changes
fail validation. Label availability is calculation time plus five minutes.

The executed sample uses January 2020–October 2024 archives. December 2019 was
unavailable; incomplete nine-settlement warm-up is excluded. There are 5,298
labels and 5,248 admitted events before evaluation exclusions. Exact 0/5/15-minute
publication assumptions preserve origin keys, settled-history features and both
validation and evaluation membership in this sample.

## Models and validation

Baselines are the current indication, last matured settlement, EWMA, training
median and a labeled +1 bp reference. Residual ridge and RF predict the realized
rate minus the current indication; a history-only ridge predicts the level.
Preprocessing is fitted on training data only, and regression targets use bp.

Initial history is 2020–2021. Monthly 2022 validation selects EWMA span 3/9/21,
ridge alpha 0.1/1/10/100 and RF depth 3/6/unlimited with leaf minimum 5/20.
RF uses 300 trees and seed 42. All candidates share 1,083 validation events per
horizon; ties follow configured grid order. Selection is frozen before scoring.

Evaluation runs from January 2023 through the final verified 18 October 2024
event. Monthly refits use labels available strictly before the fit cutoff. Whole
events are partitioned before horizon filtering; exclude decisions preceding the
fit cutoff. This excludes 12 validation and 22 evaluation boundary events.

## Experiments and inference

E1 compares models at the primary four-hour origin; E2 compares all lead times
on matched events. E3 removes feature groups and delays source availability by
one hour, reusing full-model hyperparameters. E3b scores identical forecasts
against official settlements and terminal-indication proxies. E4 describes
origin-known flat/non-flat states and calendar quarters.

Positive MAE gain means improvement over the indication. Report bp MAE, RMSE,
bias, baseline-relative squared-error skill, counts and coverage. The primary
comparison is four-hour residual ridge versus indication. Paired seven-day moving
calendar blocks give 95% intervals, with 5,000 draws, seed 42 and 3/14-day
sensitivities. Calendar gaps remain intact. Secondary model/horizon and paired
comparisons share Holm correction. State/time slices are descriptive.

Intervals are pointwise, not simultaneous bounds. Nonsignificance does not show
equivalence. The configured 0.05 bp practical threshold does not justify a
family-wide absence-of-skill claim.

## Results and limitations

On 1,889 common events, four-hour indication MAE is 0.246866 bp, ridge MAE is
0.238253 bp and RF MAE is 0.232464 bp. The primary ridge gain is 0.008613 bp with
95% interval [-0.007970, 0.028633] bp: positive primary improvement is not
established. One-hour RF gain is 0.019581 bp, secondary Holm p=0.013797.

The complete independent execution matches all 248,752 forecasts, 352 fold
records and the numerical tables within the documented tolerance. This is
same-machine computational reproduction, not independently collected evidence.
The [receipts and manifests](../results/settlement_study_v2_1/README.md) bind
sources, frozen implementation, parameters and saved outputs.

Original hourly aggregation lacks verified message-level provenance; publication
delay is assumed. Redistribution rights remain unresolved. Fresh-time/external
replication, exact closest-paper verification, DAR replication and full
seed/window/tail robustness remain outstanding. Neither forecast errors nor a
positive statistical comparison establish trading profit or market efficiency.

## Historical results

The original notebooks study distinct contemporaneous and next-row targets.
`3 * ma3 - lag1 - lag2` reconstructs the observable current indication, not the
next-row target. Current-indication persistence beats the saved original RF on
paired errors, despite both having high R². Null following-event estimates,
constant saved test columns, AIC magnitude and ADF rejection do not independently
establish forecast validity. Original metrics remain separate from settlement
results. Local notebook paths were made portable; original numerical outputs,
model artifacts and source data are retained.

## Repository contents

Tracked files include implementation, protocol configs, tests, original evidence,
completed reports and their scientific provenance. Local research planning,
implementation prompts, download probes, scratch work and operational execution
notes are not required to run the study and are excluded from Git. New experimental
outputs belong in a new versioned directory, never on top of a completed run.

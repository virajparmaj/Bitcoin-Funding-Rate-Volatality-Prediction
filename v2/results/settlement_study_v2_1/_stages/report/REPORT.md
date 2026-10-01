# Executed settlement study

Protocol: 2026-10-01-v2.1. Retrospective evaluation; original data were previously inspected.

**1889 common evaluation events** (from 1947 forecast events before the all-variant intersection), all four horizons, all models and availability variants. Primary comparison: residual ridge versus current indication at four hours.

| model | n | mae_bp | rmse_bp | gain_bp | gain_low_bp | gain_high_bp |
|---|---|---|---|---|---|---|
| ewma | 1889 | 0.286728 | 0.511293 | -0.039861 | -0.056243 | -0.024079 |
| history_ridge | 1889 | 0.293677 | 0.503406 | -0.046810 | -0.063196 | -0.031254 |
| indication | 1889 | 0.246866 | 0.470876 | 0.000000 | 0.000000 | 0.000000 |
| last_settled | 1889 | 0.307062 | 0.574579 | -0.060196 | -0.078465 | -0.043361 |
| one_bp_reference | 1889 | 0.555739 | 0.989866 | -0.308873 | -0.408715 | -0.226241 |
| rf | 1889 | 0.232464 | 0.387864 | 0.014403 | -0.001544 | 0.033304 |
| ridge | 1889 | 0.238253 | 0.384947 | 0.008613 | -0.007970 | 0.028633 |
| training_median | 1889 | 0.555739 | 0.989866 | -0.308873 | -0.408715 | -0.226241 |

Primary MAE reduction: **0.008613 bp**, 95% seven-day moving-calendar-block interval **[-0.007970, 0.028633] bp**. The primary interval does not establish a positive incremental gain.

## Secondary horizon results

| horizon | model | gain_bp | gain_low_bp | gain_high_bp | p_holm_secondary |
|---|---|---|---|---|---|
| 1 | rf | 0.019581 | 0.012179 | 0.028843 | 0.013797 |
| 1 | ridge | -0.004482 | -0.013473 | 0.005972 | 1.000000 |
| 2 | rf | 0.016637 | 0.007425 | 0.027798 | 0.137972 |
| 2 | ridge | -0.000965 | -0.012885 | 0.013159 | 1.000000 |
| 4 | rf | 0.014403 | -0.001544 | 0.033304 | 1.000000 |
| 4 | ridge | 0.008613 | -0.007970 | 0.028633 | nan |
| 6 | rf | 0.013216 | 0.003717 | 0.024921 | 0.713257 |
| 6 | ridge | -0.015428 | -0.034218 | 0.004896 | 1.000000 |

RF and non-primary horizon results are declared secondary comparisons; they do not replace the primary four-hour ridge test. Pointwise intervals and multiplicity-adjusted tests answer different questions. Statistical evidence in this retrospective sample does not establish a deployable or economically material advantage.

Positive main-variant gains passing the declared secondary Holm test: 1h rf.

## Data validation

58 verified archives; 5298 official label records; 5248 admitted events / 20992 origins before evaluation splits. 132 unusable ticker rows and 32 warm-up/eligibility origins were excluded explicitly. Maximum archive-to-schedule offset: 0.047 seconds.

0/5/15-minute publication assumptions produced identical origin keys, settled-history features and fold membership; no extra model refit was necessary.

## What was executed

- E0: full checksum-verified label history, event matching, availability checks and exclusions.
- E1/E2: monthly expanding forecasts, validation-only model selection and matched lead-time comparison.
- E3: fixed-hyperparameter indication/history/full feature fits and a one-hour source-delay variant.
- E3b: identical predictions rescored against terminal indications, plus paired differences in measured gain.
- E4: descriptive origin-known flat-state and calendar-quarter slices. These are not confirmatory subgroup discoveries.
- 3/7/14-day bootstrap sensitivity and 30/60/65-minute maximum-age sample sensitivity.

## Interpretation and limits

Positive gain means smaller MAE than the current indication. Errors are in basis points; regression error is not trading profit. Confidence intervals are conditional on the fitted forecasting procedures and do not erase retrospective protocol selection. Secondary model/horizon and paired contrasts share Holm-adjusted tests. Individual intervals are pointwise; no simultaneous practical-equivalence or family-wide unpredictability claim is made.

The one-hour delay shifts source availability, not just timestamp labels. Its staleness measures time since delayed availability; source availability, origin and variant are recorded, allowing receipt time to be reconstructed from the configured delay. Feature ablations reuse hyperparameters selected for the full feature model, so they measure fixed-configuration information removal rather than independently optimal reduced models.

Original hourly aggregation may combine fields from different messages. Its recorded arrival is an assumed conservative availability bound, not verified tick-level provenance. Label publication uses an assumed five-minute delay. Fresh-time replication, DAR replication, full seed/window/tail robustness and exact closest-paper novelty verification remain outstanding. This run does not establish publication priority, market efficiency or profitability.

The December 2019 warm-up archive returned HTTP 404; the implemented study starts in January 2020 and explicitly removes incomplete settled-history warm-up. Boundary events whose earliest origin precedes the monthly fit are excluded. The original feature file and v1 experiments are unchanged.

## Evidence

See `coverage.json`, `manifest.json`, `selection.json`, `validation_candidates.json`, `folds.csv`, `predictions.csv.gz`, `scores.csv`, `paired_contrasts.csv`, `proxy_label_scores.csv`, `state_quarter_scores.csv`, `staleness_scores.csv` and `lead_time.png`. Raw provider rows and the derived feature panel are not intended for redistribution; consult the release notes.

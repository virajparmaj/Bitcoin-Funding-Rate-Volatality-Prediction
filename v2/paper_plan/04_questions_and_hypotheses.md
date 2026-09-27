# Research question, estimands and hypotheses

## Main question

At four hours before an actual BTCUSDT funding event, does a simple residual model improve on the exchange indication available at that origin? Does incremental skill change at one, two and six hours, or between the observed +1 bp state and other states?

Let `T_s` be the scheduled event time, `F_s` its verified settled rate, and `o_s,h = T_s - h` the forecast origin. Let `q_s,h` be the last same-event indication received no later than that origin, subject to the staleness rule. Predict `u_s,h = F_s - q_s,h`, then reconstruct `F_hat = q_s,h + u_hat`.

Primary effect, in basis points:

`Delta_4 = mean(10000 * (abs(F_s - q_s,4) - abs(F_s - F_hat_ridge,s,4)))`.

Positive values favor ridge. **Residual ridge at four hours is the primary comparison.** RF is a declared secondary comparator motivated by the original project. Fixing the primary comparison avoids selecting the headline model on test performance. Hyperparameters may use 2022 validation only.

## Testable hypotheses

| ID | Hypothesis / estimand | Support needed | What weakens it |
|---|---|---|---|
| H1 | Incremental mean absolute-error improvement at four hours | Estimate and 95% paired block interval for Delta_4 | Broad interval or gain confined to one episode |
| H2 | Incremental skill varies across lead times | Paired matched-event differences in Delta_h | Differences vanish on common event coverage |
| H3 | Incremental error reduction differs between origin-known flat and non-flat states | Prespecified state interaction with event counts and uncertainty | Sparse states, unstable sign, or effect driven by a few extremes |
| H4 | Useful gains survive causal availability and information-set changes | E3 delay and feature ablations | Advantage requires later observations or different eligibility |
| H5 | Replacing actual settlements with final indications changes measured errors or model rankings | Score the same frozen predictions against each label on identical events | Differences are negligible and rankings unchanged |

The working expectation is that high next-row R² overstates incremental settlement skill. It is not a result and must not dictate data cleaning or conclusion selection.

H5 is a low-cost supporting question motivated by the measured label discrepancies. It does not require another model. It tests the consequence of a specific labeling choice, not whether the literature commonly makes that choice. If rankings do not change, report that result and retain the labels' semantic distinction without overstating its numerical impact.

## Practical negative result

Use 0.05 bp as a provisional research-resolution threshold for MAE improvement, with 0.01/0.10 bp sensitivity. It is not a profitability hurdle. Freeze the choice before the new experiment. If the upper bound for a model's improvement is below 0.05 bp, report that model-specific bound. To say neither ridge nor RF clears the threshold, use simultaneous/multiplicity-adjusted bounds for both. A nonsignificant test alone does not establish equivalence. Avoid claiming universal unpredictability.

The threshold and 65-minute age rule were proposed after inspecting historical data feasibility. This is retrospective protocol development, not preregistration of untouched evidence.

## Candidate questions retained for future work

### Eight candidate questions from the advisory assessment

Novelty judgments below are qualitative, conditional on the accessible literature; they are not publication probabilities.

**RQ1 — Does a regression model improve the final BTCUSDT funding forecast over the exchange indication observed four hours before settlement?** Hypothesis: the saved level-score advantage will not translate into a stable advantage over that stronger baseline; any useful residual signal may depend on lead time. Why it matters: it measures information the market has not already supplied. Prior work covers funding predictability; exact overlap at this information set remains to be verified. Existing support: ticker trajectory plus the corrected saved-output comparison. Additional work: authoritative labels, as-of panel, residual ridge/RF models and walk-forward inference. Distinctness: moderate, strongest candidate. Risk: historical availability cannot be reconstructed or the closest paper already tests it.

**RQ2 — How does forecast skill change at one, two, four and six hours before the same settlement?** Hypothesis: the indication's error generally falls as the event approaches, while model improvement is less stable than level R² suggests. Importance: forecast usefulness depends on advance notice. Prior literature explains funding calculation and dynamics; a monotonic improvement is not itself novel. Existing support: 69.7% of event groups show within-event revisions. Additional work: matched event set, separate horizon models and paired curve differences. Distinctness: moderate as part of RQ1. Risk: different missing-data coverage rather than lead time explains the curve, or revisions are not monotonic.

**RQ3 — Does +1 bp state persistence obscure model performance away from that state?** Hypothesis: models perform differently when the observable indication is at the interest anchor versus outside it. Importance: average errors can hide where decisions are hard. Funding-rule flat regions are established; their predictive interaction remains an empirical question here. Existing support: 41.0% of rows at +1 bp. Additional work: define states at the origin, test residual loss by state, and evaluate transition probabilities with tie-aware labels. Distinctness: moderate-to-low alone. Risk: too few transitions, mechanical classification, or historical rules differ from current documentation.

**RQ4 — How much does using nominal row time instead of available-at time change measured skill?** Hypothesis: timing conventions materially change features, origins or rankings. Importance: end-of-hour values can be improperly timestamped at the beginning of the hour. Prior literature establishes look-ahead bias; this would quantify a funding-specific instance. Existing support: synthetic clock and current ingestion code. Additional work: aligned clean evaluation, one-at-a-time timing perturbations and conservative one-hour delays. Distinctness: low-to-moderate supporting result. Risk: an idiosyncratic archive defect, not a general phenomenon.

**RQ5 — Does retaining extreme training observations improve tail forecast error without materially harming ordinary-period accuracy?** Hypothesis: deleting/zeroing extreme funding values harms stress performance. Importance: stress episodes are economically consequential. Robust/tail modeling is established. Existing support: 1,118 zeroed observations. Additional work: compare unmodified outcomes with train-only robust fitting and legacy cleaning, on identical untouched test outcomes. Distinctness: low alone, useful ablation. Risk: no improvement or high variance from few tail events; do not force the desired conclusion.

**RQ6 — Can past uncertainty predict the magnitude of the remaining settlement surprise better than a rolling residual-variance baseline?** Hypothesis: variance models help predict surprise size even when mean forecasts add little. Importance: uncertainty may matter for risk decisions. Funding heteroskedasticity/GARCH and jumps have precedents. Existing support: GARCH code, but no valid out-of-sample variance evidence. Additional work: sequential forecast residuals, fixed mean forecast, positive variance forecasts, QLIKE or proper density/interval scores. Distinctness: moderate if tied to lead-time residuals. Risk: noisy variance proxies, numerical instability and a larger study than intended.

**RQ7 — Are any incremental gains stable under rolling versus expanding estimation and across later calendar blocks?** Hypothesis: model ranking depends on estimation window and market state. Importance: guards against an isolated favorable holdout. Time variation is already established in related funding research. Existing support: multi-year data and qualitative early-versus-late observations. Additional work: common-origin rolling/expanding reruns, quarterly loss differences and uncertainty intervals. Distinctness: low alone; essential robustness. Risk: hindsight in window selection or inadequate power within quarters.

**RQ8 — Does a forecast-guided funding policy add net value over the identical policy driven by the exchange indication?** Hypothesis: forecast accuracy gains need not translate into incremental net returns. Importance: separates a statistical predictor from an actionable decision. Funding-arbitrage and funding-aware control are established. Existing support: prices, indicative rates, proposed fees, no execution history. Additional work: settled cash flows, tradable two-leg prices, borrowing/collateral assumptions and turnover-aware cost sensitivity. Distinctness: low-to-moderate supporting contribution. Risk: unavailable execution data and a policy chosen to fit the test set. Defer from the minimum paper.

Priority by defensibility and reuse: **RQ1 + RQ2**, supported by **RQ3/RQ4/RQ7**. RQ5 is inexpensive after the pipeline works. RQ6 or RQ8 should be separate extensions unless the core result specifically requires them.

# Research assessment: Bitcoin perpetual funding forecasts

Prepared 25 September 2026. Scope: the repository, its locally available dataset and saved outputs, and the literature accessible during this review. This is an advisory assessment and a proposed study, not a completed paper or a claim of publication readiness.

**Recommendation:** develop a narrowly scoped empirical paper on **forecast skill beyond the exchange's current funding estimate, measured at explicit lead times before settlement**. Use the existing project as a motivating reproducibility case. Do not build the paper around a novel RF–GARCH pipeline, a 97.5% forecasting score, or the assertion that funding-rate prediction has not been studied.

The most useful result already present is that the saved Random Forest loses to correctly timed persistence. More importantly, the existing v2 audit misidentifies both the prediction target and the meaning of the provider's funding fields. Correcting those interpretations makes a better research question possible with much of the data already available.

## 1. Reconstructing the actual investigation

### Materials and verification scope

Reviewed all four notebooks, their stored text outputs, preprocessing and model utilities, ingestion code, v2 audit helpers/configuration/tests, README files, the normalized data, and the saved prediction CSVs. Saved pickle files were inventoried but not executed. The original report linked through Canvas is not locally present; its contents were not available. Claims attributed to absent `notes/` files in the existing audit cannot be independently verified from those files.

The locally supplied data contain **42,179 rows and 11 columns**, Binance BTCUSDT only. Nonmissing message-arrival timestamps span **2020-01-01 00:59:57 UTC through 2024-10-17 23:59:59 UTC**. There are **5,256 distinct upcoming funding-event timestamps**. These are event identifiers, not 5,256 independently verified settled rates. The dataset is an hourly-style reduction of derivative-ticker messages, not a transaction tape, order book, or authoritative settlement ledger.

The new [evidence script](reproduce_evidence.py) rereads the input files, verifies row/target alignment, recalculates the headline scores, records input SHA-256 hashes, and creates [machine-readable measurements](evidence.json), a [score table](saved_prediction_scores.csv), and a [diagnostic figure](baseline_comparison.png). It does not rerun the original model training or claim to establish clean out-of-sample performance. Input hashes were unchanged after execution.

### Three different tasks were mixed together

| Component | What the code actually investigates | What it does not establish |
|---|---|---|
| Analysis A, `stat429_analysis_a.ipynb`, JSON cells 22–30 | Contemporaneous funding-rate regression with linear regression and RF, using shuffled splitting and a rolling standard deviation containing the response | A future funding forecast, temporal generalization, or a clean nonlinear-model advantage |
| Model 1, `model_development.ipynb` | Classification of whether the next row's rate is higher; ties belong to class 0 | Prediction of price direction; three-way up/flat/down skill; settlement-level classification |
| Model 2, JSON cell 11 | In-sample GARCH-family model fitting and information-criterion selection, followed by a terminal variance forecast | Out-of-sample variance accuracy; improvement from Model 1; calibrated uncertainty |
| Model 3 RF, JSON cell 14 | Prediction of `funding_rate.shift(-1)` from present/past row features | A forecast of the next payment at a fixed advance notice |
| Model 3 SARIMAX, JSON cell 17 | A block of out-of-sample predictions with test-period exogenous features and no sequential target-state updating | An apples-to-apples rolling one-step comparison |
| v2 | An implemented historical audit plus plans for corrected experiments | Completed settlement construction, baselines, validation, walk-forward tests, or trading evaluation |

`models/model2.py` and `models/model3.py` are empty. The actual research logic resides in notebooks. Stored notebook outputs include a pipeline error for `keep_future_rate`; current code and saved execution history are not identical. Calls to `add_model2_volatility(..., steps=5)` also disagree with the current helper signature. A clean-kernel reconstruction is necessary before attributing saved predictions to the current source version.

### A key correction: an observable current rate is not a leaked future target

Let `q_t` denote the funding rate displayed in row t. The model predicts `q_(t+1)`. Its features contain

`3 × ma3_t − lag1_t − lag2_t = q_t`.

This reconstructs the **current observation**, not `q_(t+1)`. If row t is available at the stated forecast origin, using it is legitimate. Algebraic recoverability of an observed current value does not establish leakage of an unobserved future value.

I joined saved RF rows back to the raw file using the synthetic timestamp solely as a row key. Among **7,793 rows with both raw current and next values present**, the reconstruction matches the current rate to numerical precision, and `Actual` matches the next raw rate exactly. Four rows lack one of those raw values. This independently confirms the timing interpretation on the verifiable subset.

On all 7,797 saved RF test rows:

| Predictor | R² | MAE, bp | RMSE, bp |
|---|---:|---:|---:|
| Current-rate persistence, recovered from features | 0.980076 | 0.094347 | 0.194399 |
| Saved Random Forest | 0.974973 | 0.140612 | 0.217880 |
| EMA3 | 0.955758 | 0.145388 | 0.289684 |
| `funding_rate_lag1`, an older observation | 0.936958 | 0.170895 | 0.345801 |

Thus the RF has **25.62% greater MSE and 49.04% greater MAE than current-rate persistence**. Its squared-error skill relative to that baseline is **−0.2562**, despite R² ≈ 0.975. The older lag column used as the audit's headline baseline is unnecessarily stale for this task.

These are descriptive comparisons on the saved artifacts. They do not establish that all RF models fail, that markets are efficient, or that there is no profitable funding strategy.

### A second correction: the exchange estimate is probably already in the file

The provider distinguishes the upcoming event's updating rate from an estimate for the event after that. `funding_rate` belongs to the immediately upcoming `funding_timestamp`; `predicted_funding_rate`, when supplied, refers to the following event. A null latter column does not imply absence of an upcoming-event estimate. See the provider's [derivative-ticker schema](https://docs.tardis.dev/downloadable-csv-files/data-types.md#derivative_ticker).

This is consistent with direct measurement: **3,664 of 5,256 event groups (69.71%) contain multiple distinct funding rates**. These are not simply eight repetitions of an already realized settlement. The mean absolute first-to-last rate change within a group is **0.6225 bp**, a descriptive measure of revisions, not forecast skill.

The exact historical Binance-to-Tardis mapping should still be archived and verified against settlement records. A group's final hourly observation is a **terminal indicative-rate proxy** until that verification succeeds. The final observation may be stale or slightly after the scheduled event in arrival time. Grouping by event and taking `last()` alone is insufficient.

### What is demonstrated, suggested, and unknown

**Demonstrated from current files or direct recomputation:**

- Saved RF predictions lose to current-rate persistence on the same saved rows.
- SARIMAX's saved 7,622-row output has **R² = −4.807223**. If the documented 10⁶ scaling applies, its MAE is **1.7871 bp**, not the enormous bp value printed by the audit's unscaled scoring helper.
- Preprocessing backfills **8,531 cells**: 5,351 index-price cells and 3,180 open-interest cells. This uses future observations; the magnitude of its effect on model performance is unmeasured.
- The legacy outlier-plus-zero-fill path replaces **1,118 funding observations** with zero; their original mean absolute magnitude is **12.4411 bp**.
- The synthetic clock ends about **130 hours ahead** of arrival time. A nominal test start in December 2023 maps to a real start of **28 November 2023**.
- Saved test columns for Model 1 direction and Model 2 volatility are constant. There is no demonstrated time-varying contribution from those inputs on this test set. Constant test values alone do not prove the columns had no effect during training.
- About **41.00% of all raw rows equal +1 bp**, inviting a mechanism-aware baseline and flat-state analysis.

**Strongly suggested, but not yet demonstrated as a research conclusion:**

- Much of the high next-row R² reflects predictable persistence in the displayed estimate.
- A fixed time before settlement may be more informative for evaluating a forecasting system than an arbitrary next-row horizon.
- Apparent direction accuracy may be sensitive to the treatment of unchanged rates.
- Signal and failure rates may differ between the +1 bp state, non-flat states, and extreme episodes.

**Not demonstrated:**

- Reliable improvement over the exchange estimate at any settlement lead time.
- Reliable prediction of settled-rate innovations, future volatility, or economically valuable decisions.
- Statistical significance after serial dependence and repeated model/horizon comparisons.
- Generalization to another asset, venue, later period, or a deployable execution setting.
- A contribution from a correctly constructed stacked direction/volatility pipeline.

### Additional corrections required before reusing v2 claims

1. **The clock defect's mechanism is unresolved.** No adjacent nonmissing arrival-time difference exceeds 90 minutes; some differences are subsecond. The existing explanation of dropped days causing all the drift is not supported by this diagnostic. Near-hour-boundary arrivals also make naive flooring misleading. Preserve actual arrival times and audit the original sampling procedure.
2. **Analysis A's repair ladder is confounded.** Random and chronological splits use different test distributions. Changing the target changes the task. The observed decline cannot be partitioned into causal contributions of individual defects. Its stored audit values (0.9357, −3.2385, −6.4664, −6.2178) are recorded audit outputs, not rerun model results from this review.
3. **R² = 0 is the evaluation-set mean benchmark**, not generally the predictor zero. Negative R² does not by itself prove an absence of trading value.
4. **An ADF rejection does not prove globally stable distributions**, constant variance, or that every differenced forecasting model is invalid. Compare justified specifications prospectively rather than declaring automatic over-differencing from one test.
5. **AIC magnitude depends on units; relative AIC on identical samples and units remains meaningful.** For identical likelihoods under rescaling, all AIC values shift by the same constant. This does not invalidate their ranking. The GARCH cell calls the default unscaled pipeline, so the audit's statement that this particular fit used 10⁶ scaling is not supported by that cell. Absolute GARCH coefficients are not comparable feature-importance scores.
6. **SARIMAX uses supplied future-block exogenous values.** It is not simply an unconditional forecast with no new information. Whether each exogenous row is admissible depends on the claimed origin. The state-update/horizon inconsistency remains real.
7. **MAE improvement is not profit.** Dividing an MAE difference by a fee and calling the ratio economic unviability is invalid. A policy, its turnover, holding duration, funding cash flows and price/basis exposure must be specified. The repository's fee constants are scenarios, not verified account-specific historical fees.
8. **The AR(1) effective-n approximation is not a universal sample size.** It approximates uncertainty for particular statistics under assumptions; it cannot replace the sample size for every regression or forecast test.
9. **Eight settlements are not one day.** At three eight-hour events per day, an HAC lag of 8 spans about 64 hours. The comment in `v2/src/config.py` must not determine the inference design.
10. **Feature validity is about availability.** A blanket ban on timestamps, constants, or reconstructing the current rate is not a leakage detector. Time-to-settlement is a valid, essential feature; a constant can be a meaningful benchmark. Missingness/staleness flags must only describe information known at the origin.

## 2. Closest research and limits of the literature search

Searches covered arXiv, publisher-hosted journal literature, SSRN working papers, official conference material, and methodological references, emphasizing 2021–2026 and retaining the directly relevant 2019 funding/GARCH paper. Queries included funding-rate prediction/forecasting, RF, LSTM, persistence, nowcasting, lead time, settlement, and leakage, followed by title searches and citation tracing.

**Access limitation:** direct Google Scholar and Semantic Scholar search pages were inaccessible. Domain-restricted web searches were attempted but did not provide dependable complete results. This is a substantial targeted search, not an exhaustive systematic review of those indexes. SSRN full text for the closest Inan paper returned an access error/403; its author abstract and official conference abstract were available. Do not turn absence from accessible abstracts into a claim that a paper omitted an experiment.

### Important papers

**1. Emre Inan, “Predictability of Funding Rates” (2025 working paper; CFE-CMStatistics 2025 presentation).**

Question/method/data: one-step out-of-sample prediction of Bitcoin perpetual funding on Binance and Bybit using double autoregressive models and standard benchmarks. Reported result: improvements over no-change forecasts in error and direction, with time-varying predictability. Exact sample dates, effect sizes and full comparison set could not be verified. Your generic forecasting and regime questions overlap directly. A potential distinction is fixed within-settlement lead times and incremental skill against the contemporaneous exchange indication; whether the full paper already covers this must be checked before claiming novelty. [Author abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424); [official conference abstract](https://www.cmstatistics.org/RegistrationsV2/CFECMStatistics2025/viewSubmission.php?in=1301&token=o1n4r30pp8r239sq1o0os653591o7nn1).

**2. Sai Srikar Nimmagadda and Pawan Sasanka Ammanamanchi, “BitMEX Funding Correlation with Bitcoin Exchange Rate” (2019 arXiv).**

Question: funding heteroskedasticity and its relation to Bitcoin prices. Methods: ARCH, ADF, Granger tests, GARCH-family selection. Data: 3,649 eight-hour observations, BitMEX funding and Bitstamp price, June 2016–October 2019. Reported result: EGARCH(1,1) is preferred by information criteria. The inspected paper does not provide the proposed lead-time/exchange-estimate benchmark. It substantially precedes your GARCH idea; changing venue or selecting GJR rather than EGARCH is weak novelty. Your opportunity is properly timed forecast evaluation. Granger predictability should not be repeated as structural causality. [Paper](https://arxiv.org/html/1912.03270).

**3. Shreyash Kharat, “Stochastic Modeling of Funding Rate Dynamics with Jumps” (2025 SSRN).**

Question: can mean reversion with jumps represent BTCUSDT funding dynamics? Uses Binance minute data for 17–24 December 2024 and a 25 December test, OU models, simulation, MCMC and jump detection. The abstract reports unexplained variance under OU and computational difficulties motivating a faster jump procedure. Your volatility/tail interests overlap; a multi-year evaluation of settlement-surprise errors would differ from this short-sample stochastic fit. Exact forecast gains and omitted experiments were not verified from full text. [Author abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5290137).

**4. Songrun He, Asaf Manela, Omri Ross and Victor von Wachter, “Fundamentals of Perpetual Futures” (2022 initial draft; September 2026 version inspected).**

Question: pricing and arbitrage with funding and trading frictions. Uses theoretical bounds and empirical crypto spot/perpetual data, including Binance contracts from 2020 through March 2024 where available. Reports substantial pricing deviations, declining deviations over time, and attractive simulated arbitrage performance. This establishes that funding mechanics and clamps matter; those are not new discoveries for your paper. Its primary estimand is pricing/arbitrage, not the proposed incremental forecast comparison. Different paper versions must be cited explicitly. [Version inspected](https://arxiv.org/html/2212.06888v7).

**5. Damien Ackerer, Julien Hugonnier and Urban Jermann, “Perpetual Futures Pricing” (2023 preprint; Mathematical Finance, 2026).**

Question: no-arbitrage pricing of linear, inverse and quanto perpetuals. Primarily analytical models in discrete and continuous time; derives pricing expressions and funding specifications supporting spot anchoring and replication. No empirical forecast-training dataset is central to the result. Your work could test empirical predictability under an actual information set, not claim new pricing theory. [NBER record and published-version details](https://www.nber.org/papers/w32936); [arXiv](https://arxiv.org/abs/2310.11771).

**6. Petar Zhivkov, “The Two-Tiered Structure of Cryptocurrency Funding Rate Markets” (Mathematics, 2026).**

Question: cross-venue integration and exploitable funding spreads. Uses correlations, time-series econometrics, Granger tests and cost accounting on 35.7 million minute observations, 749 symbols and 26 venues, 8–15 November 2025. Reports stronger CEX integration and only 40% of selected top opportunities profitable after costs/reversals. Your cross-venue and economic ideas overlap. Its short cross-sectional sample differs from your multi-year single-contract lead-time study. Do not claim cross-exchange dispersion or cost-aware evaluation itself as new. [Journal article](https://www.mdpi.com/2227-7390/14/2/346).

**7. Warodom Werapun, Tanakorn Karode, Jakapan Suaboot, Tanwa Arpornthip and Esther Sangiamkul, “Exploring risk and return profiles of funding rate arbitrage on CEX and DEX” (online 2025; 2026 journal issue).**

Question: funding-arbitrage returns, leverage and diversification across Binance, BitMEX, ApolloX and Drift, with BTC/ETH/XRP/BNB/SOL. Compares 60 scenarios and HODL; reports a best six-month return of 115.9% with 1.92% drawdown for leveraged Drift XRP. These are their backtest claims, not independent validation. The forecast baseline/lead-time question is separate from their strategy comparison. Your project cannot infer profitability from prediction errors or claim to introduce funding arbitrage. [Journal article](https://doi.org/10.1016/j.bcra.2025.100354).

**8. Nam Anh Le, “Funding-Aware Optimal Market Making for Perpetual DEXs” (2026 arXiv).**

Question: how stochastic funding changes inventory/quotation control. Uses HJB numerical methods, OU funding and Hyperliquid BTC/ETH/SOL calibration, with 100-seed holdout simulations. Reports improved mean BTC/ETH performance and lower inventory RMS against a classical market-making baseline; SOL is less robust to a risk-scaled comparison. It evaluates a control policy, not your proposed realized-settlement prediction benchmark. It reinforces the separation of forecast metrics from decision value. [Paper](https://arxiv.org/abs/2605.06405).

**9. Hansika Hewamalage, Klaus Ackermann and Christoph Bergmeir, “Forecast evaluation for data scientists: common pitfalls and best practices” (2023).**

Question: when forecast evaluation gives misleading rankings. A methodological tutorial with illustrative experiments, not a crypto benchmark, covering persistence, metrics, partitioning and leakage. It shows that inappropriate evaluation can favor apparently sophisticated models. Your audit repeats established principles; a contribution requires a domain-specific measurement and empirical result, not another generic warning against shuffled splits. [Journal article](https://doi.org/10.1007/s10618-022-00894-5).

**10. Sayash Kapoor and Arvind Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science” (Patterns, 2023).**

Question: how leakage compromises scientific claims. Uses a cross-field review and a civil-war-prediction replication, showing that apparent complex-model advantages can disappear under corrected evaluation. This is methodological precedent for your case, not evidence of leakage in any particular funding paper. Your distinct contribution would need a precise funding-information protocol and results beyond one broken notebook. [Journal article](https://doi.org/10.1016/j.patter.2023.100804).

**11. Sicheng He, Shirui Wang and Tianyang Zhang, “A Shared Template Without Shared Feedback: Funding Rates in Cryptocurrency Perpetual Futures” (2026 SSRN, revised September).**

Question: why common nominal funding rules need not deliver common stabilization. Combines a constrained-arbitrage model with evidence from 200 Binance perpetuals; the abstract emphasizes interval averaging, flat/capped regions, liquidation and pre-settlement behavior. This overlaps a mechanistic interpretation of flat states. It does not establish the result you seek; its full benchmark coverage was not accessible. Your possible distinction is forecast-error measurement at admissible origins, not discovering clamps or settlement timing. [Author abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6185958).

### Conference, benchmark and citation screening

The official CFE-CMStatistics presentation is directly relevant. A 2025 AI-in-Finance conference PDF search result also describes funding-based predictors of **spot realized volatility**; full text could not be retrieved, so its title/authors/results are not used as verified evidence here. That target differs from funding-rate volatility. A UAI 2021 paper on Fisher-information “data leakage” was screened out because it concerns privacy leakage, not future-information contamination. Avoid keyword-based citation padding.

A later article cites “Forecasting cryptocurrency perpetual swap contract funding rates” as Akyildirim et al. (2021). I could not locate an independent primary record for that exact reference. Treat it as an unresolved citation lead, not an established competitor or an invented bibliographic entry. Resolve it during the final novelty check.

No standardized public funding-forecast benchmark was verified in this search. That does not establish that none exists. Inan is the first empirical comparator to obtain and reproduce. General time-series benchmarks cannot validate a funding-specific horizon or field interpretation.

**Answer to “has essentially the same study been published?”** The broad funding-predictability question, GARCH fitting, regime variation, and funding-arbitrage economics already have close precedents. I did not verify a paper with the exact proposed lead-time/current-indication experiment, but incomplete full-text access prevents a “first study” claim. The current project is not yet distinct enough as a standard model comparison.

## 3. Novelty-gap analysis

| Candidate contribution | Defensibility | Work reuse | Principal objection |
|---|---|---|---|
| Incremental settlement forecasting beyond the current exchange indication, by lead time | **Best candidate; novelty provisional** | High: current ticker, prices, OI, regression infrastructure | Full-text prior-art overlap; settlement-label and timestamp integrity |
| Empirical contrast between next-update forecasting and settled-rate forecasting | Good supporting contribution | Very high | Different targets cannot be described as a controlled accuracy improvement |
| Flat-state versus non-flat-state forecast behavior | Useful secondary question | High | Funding clamps/flat regions are already known; retrospective regimes can leak |
| Effect of timestamp and availability conventions on rankings | Useful reproducibility result | High | A single implementation error is not a general scientific finding |
| Tail-event failure and robust estimation without deleting true outcomes | Useful secondary question | High | Few independent extremes; thresholds chosen after seeing outcomes |
| Predicting residual uncertainty rather than the funding level | Potentially useful, more work | Medium: GARCH ideas reusable | Variance target and evaluation require a new, careful definition |
| Out-of-time stability and rolling versus expanding fits | Necessary credibility; weak standalone novelty | High | Closely related work already treats time-varying predictability |
| Cross-exchange residual information at fixed origins | Potentially stronger extension | Low with current data | New timestamp alignment, data access and venue-specific mechanisms |
| Lightweight model accuracy/runtime trade-off | Low standalone novelty | High | One small dataset and ordinary models make the result unsurprising |
| RF–GARCH–direction stacking | Currently unsupported | Low-to-medium | Must generate genuinely out-of-fold intermediate predictions; prior model combinations are numerous |
| Generic leakage detection, SHAP rankings, new asset/model combination | Insufficient alone | High | Known techniques and correlation do not create a research contribution |

A reusable benchmark becomes credible through an unambiguous target, available-at timestamps, public label provenance, fixed splits, meaningful baselines and released forecasts. Calling an ordinary dataset “new” is insufficient. Correlated tree importance, or a GARCH coefficient bar chart, does not demonstrate a mechanism. Use retrained feature-group ablations instead.

## 4. Eight candidate research questions

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

## 5. Minimum publishable extension: exact experiments

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

## 6. Hidden results and possible central stories

The strongest existing hidden result is **baseline misalignment reverses the apparent model advantage**. The discovery is not that a linear identity predicts an unknown target; it is that the features already encode the right no-change predictor, while the audit compares against an older one. Use this to motivate the paper and to explain why R² alone answers the wrong question.

The second is **a confusion between an evolving indication and the settled payment**. That permits a substantive distinction between predicting the next update of a public estimate and predicting what will actually be paid at an actionable horizon. This can be central if verified labels and E1–E3 yield a stable quantitative conclusion.

The third is **the visible flat state**. A high fraction at +1 bp is compatible with the exchange's known interest/clamp mechanism, not proof of duplicate data or an original discovery. The research opportunity is a measured difference in incremental prediction errors and transitions conditional on the state. [Exchange mechanism](https://www.binance.com/en/support/faq/detail/360033525031).

Other observations are useful but insufficient as the main story: an implemented stack with constant saved outputs; loss of extremes through cleaning; stale notebook/source interfaces; and inconsistent scaling in result reporting. An early-versus-late volatility change is currently qualitative. LR's claimed 80.26% accuracy versus RF's 78.75% and RF's larger F1 are stored narrative claims without saved classification predictions here; they suggest metric trade-offs, not a verified publication result. Seed instability, clean model disagreement, and computational trade-offs have not been measured.

## 7. Strongest defensible research argument

**Using only results available now:**

> Previous research has established that perpetual-futures funding has structured dynamics and has been studied with statistical forecasting models.
>
> However, it remains unclear from the current project's evaluation whether its high next-row scores represent information beyond an observable current funding indication, or accuracy at a defined time before payment.
>
> We investigate this distinction using the project's Binance BTCUSDT ticker history, source code and saved predictions.
>
> Our experiments evaluate target alignment, field semantics and simple baselines on the saved test rows.
>
> Our results show that the saved Random Forest has 25.6% higher squared error than current-rate persistence despite an R² near 0.975; the data also contain changing within-event indications rather than only repeated realized rates.
>
> These findings suggest that the original performance numbers do not establish incremental settlement-forecasting skill.
>
> The contribution of this work at present is a reproducible diagnostic case and a precise specification of the additional evaluation required, not a validated new forecasting method.

For the completed paper, replace the fourth through seventh paragraphs with E1–E4 results: the exact lead-time gains and intervals, observable states in which they persist, and a narrow single-venue claim. **Do not prewrite “no predictability” or “profitable forecasts” as the conclusion.** If the corrected model helps, report the improvement. If an upper confidence bound rules out a meaningful gain, report that bound. If intervals are broad, the scientific result is inconclusive and more data may be necessary.

## 8. Completed study design

| Item | Proposed specification |
|---|---|
| Research question | Does observable market information improve the next settled BTCUSDT funding prediction beyond the current exchange indication, and how does this depend on lead time? |
| Primary hypothesis | High next-update R² overstates the model's incremental information relative to current-indication baselines; evaluate the actual gain rather than assuming its sign |
| Secondary hypotheses | Lead time and origin-observable flat/volatile states change forecast errors; extra market features may help only in some states |
| Dataset | Existing 2020–October 2024 ticker reduction plus authoritative settled labels; preserve original file and provenance |
| Unit of analysis | A funding event at a specified origin; repeated horizons clustered within events and calendar blocks |
| Experimental design | Locked retrospective expanding-window evaluation, group-consistent splits, purged label overlap, shared eligibility; later untouched replication if available |
| Models/baselines | Current indication, last settled rate, EWMA, median/+1 bp references, low-order history model, residual ridge and RF; DAR replication where feasible |
| Evaluation | Primary MAE gain in bp; RMSE, relative squared-error skill, signed bias, coverage, counts; optional proper uncertainty scores |
| Statistical testing | Paired calendar-block bootstrap; HAC/DM sensitivity; Holm correction; equivalence/practical-gain bound for negative claims |
| Robustness | Rolling/expanding fits, availability delays, sample intersections, staleness thresholds, years, seeds, missingness; no outcomes replaced by zero |
| Ablations | Indication, indication history, settled history, then price/OI groups; train-only robust fitting; sequentially valid stack only if retained |
| Primary result | A lead-time-specific estimate and interval for incremental forecast skill; currently unavailable |
| Secondary results | Baseline reversal in saved artifacts; flat-state/tail localization; stability over time; calibration or cost results only if evaluated |
| Figures | Information timeline; lead-time skill curve; cumulative paired loss or quarterly/state comparison; diagnostic saved-baseline figure in appendix |
| Tables | Data/availability coverage; literature/target comparison; principal clean leaderboard; ablations and robustness |
| Limitations | One instrument/venue, historical retrospective design, coarsened asynchronous ticker fields, unavailable original tick tape, limited extremes |
| Threats to validity | Label proxies, arrival-time ambiguity, test reuse, post-selected stress periods, historical rule changes, missing-data selection, weak baselines, model selection, dependence and tail-sensitive inference |
| Reproducibility | Versioned inputs/hashes, field/units manifest, event IDs and origins, forecasts per event, fixed configurations/seeds, locked dependencies and one-command execution |

The code organization should reuse v2's additive separation but revise its scientific assumptions. A compact implementation needs (a) event/availability/label construction, (b) baseline/residual features and models, (c) split/inference helpers, and (d) a thin experiment runner. Existing `audit.py` and `legacy.py` remain historical evidence helpers. The features themselves can be reused conceptually; the preprocessing implementation should not be reused blindly.

Store predictions with `event_id`, `settlement_time`, `origin_time`, `source_available_at`, `horizon`, `age`, `fold`, `model`, `target`, `prediction`, `units`, `feature_version` and `train_end`. Assert `max(feature_available_at) <= origin_time` and `train_label_available_at < fit_time`. Preserve the current indication as an explicit field. Test known violations, boundary events and unit conversions, not merely function names or string bans on `bfill`.

Because `resample(...).last()` can choose the last nonmissing value separately by column, a reduced row can combine asynchronously observed fields. Conservatively using the row's final arrival time bounds availability only if the upstream construction is valid. Without raw-message provenance, report this assumption and a lag sensitivity rather than certifying exact point-in-time reconstruction. Never derive forecast features from the eventual last observation within the event.

Data release is also part of feasibility: confirm redistribution rights for provider-derived data; if redistribution is unavailable, release acquisition instructions, hashes, code and permitted derived predictions. Do not promise an open benchmark based on data others cannot legally obtain.

## 9. Paper structure and the evidence each section needs

| Paper section | Argument and required evidence |
|---|---|
| 1. Abstract | State target, origin, baseline, sample and principal gain with interval. Distinguish observed findings from planned work. No percentage “accuracy” derived from R². |
| 2. Introduction | Motivate an upcoming cash-flow forecast. Show how a high level score can coexist with negative incremental skill. Define the specific unresolved information-set question. |
| 3. Related Work | Separate funding dynamics, pricing/arbitrage, and forecast-evaluation methods. Compare exact targets/origins/baselines with the closest forecasting study after full-text verification. |
| 4. Research Question / Hypotheses | Predeclare the four-hour primary comparison, secondary horizons, practical-gain threshold and falsifying outcomes. |
| 5. Data | Provider field semantics, true time span, availability assumptions, final-label source, gaps/staleness, historical schedule and +1 bp state. Include the sample-flow table. |
| 6. Methodology | Define F_s, q_(s,h), residual targets and as-of joins. Specify what is known at every origin. Explain why using the current indication is legitimate. |
| 7. Experimental Setup | Train/validation/test dates, monthly updating, purge rules, eligibility, feature groups, models, tuning budget, loss functions and dependence-aware inference. |
| 8. Results | Lead with the matched-event skill table and horizon plot; include uncertainty and all baselines. Keep legacy diagnostic results separate. |
| 9. Robustness / Ablations | Report timing delays, train-only cleaning, feature groups, origin-known states, alternative windows and seed sensitivity. Explain selection/coverage changes. |
| 10. Discussion | Explain what information is already embodied in the indication and where residual uncertainty remains. Discuss implications for risk decisions without asserting untested profits. |
| 11. Limitations | Admit retrospective test reuse, one venue/asset, coarse provenance, possible prior-art overlap, modest sample of extremes and unmodeled execution. |
| 12. Conclusion | Answer the declared question within this sample and protocol. State positive, bounded-negative or inconclusive results faithfully; identify one next replication. |

Appendices: legacy claim reconciliation, exact configurations, every saved forecast, alternate block lengths, sensitivity results and provider/exchange schema snapshots. A main paper should contain one coherent empirical question; it need not retain every model from the coursework.

## 10. The paper's central story and readiness decision

**Proposed title:** *Beyond Persistence: Evaluating Bitcoin Funding Forecasts Against Exchange Indications at Fixed Settlement Lead Times*.

**One-sentence contribution:** A reproducible, event-aligned evaluation of whether simple statistical and tree models add information beyond the exchange's currently published funding indication, and where that information survives changes in lead time and market state.

**Research gap:** the incremental information question under a precisely timed, event-specific baseline; its distinctness remains provisional until the closest full texts are checked.

**Primary question:** at a four-hour advance notice, can a residual model materially improve prediction of the actual settlement rate over the observed indication?

**Primary hypothesis:** high next-update R² can coexist with negligible or negative incremental settlement skill. The first half is demonstrated in the saved diagnostic; the settlement claim remains untested.

**Novelty claim:** a domain-specific empirical benchmark and lead-time/state result, not a new algorithm, a new funding mechanism, or the first funding forecast.

**Most important experiment:** E1/E2, scoring identical realized events against a contemporaneously available exchange indication at fixed horizons.

**Most important result needed:** a stable effect estimate with intervals that supports either meaningful incremental skill or a sufficiently tight bound excluding a prespecified meaningful improvement. A failed significance test or another high R² is insufficient.

**Minimum additional work:** verify labels/timing and revise the audit interpretation; run E0–E3; obtain the closest paper's full methodology; publish a reproducible evidence package. No new neural network, cross-exchange feed, or trading system is required for this minimum scope. Exact effort depends on recovering trustworthy labels and availability; that is the uncertainty to resolve first.

**Potential abstract, restricted to what is currently known:**

> Funding-rate forecasting studies must distinguish predictions of an updating exchange indication from predictions of a realized settlement payment. We examine a Binance BTCUSDT research pipeline containing 42,179 ticker-derived observations from January 2020 to October 2024. An audit of 7,797 saved next-observation predictions finds that a Random Forest achieves an R² of 0.975 but has 25.6% higher squared error and 49.0% higher absolute error than a current-rate persistence baseline. Feature identities reconstruct the current observed indication rather than the future target, changing the interpretation of a previously alleged leakage mechanism. The data also show within-event variation in 69.7% of funding-event groups, underscoring the distinction between indicative updates and realized payments. We specify an event-aligned evaluation using actual settlement labels, admissible feature times and exchange-indication baselines at fixed lead times. The present results establish a reproducibility case; incremental settlement-forecasting performance remains to be measured.

This is a diagnostic-study abstract, **not a ready abstract for the proposed completed empirical paper**. After E0–E3, replace its last two sentences with the measured horizon-specific estimates, intervals and restricted conclusion. Do not submit a future-experiment plan as a completed empirical result.

**Three strongest planned displays:**

1. A lead-time figure showing both absolute forecast error and incremental gain over the exchange indication, on matched settlements, with uncertainty bands.
2. A main table comparing current indication, settled persistence, EWMA, linear and RF residual models with MAE, RMSE, gain intervals and sample counts.
3. An observable-state/quarter stability display identifying whether gains survive outside flat periods and isolated extremes.

The saved-output baseline-reversal figure already exists and is appropriate as motivation or an appendix, not a substitute for these displays.

**Closest competitors:** Inan for direct forecasting, Kharat for stochastic funding dynamics, and the 2019 BitMEX paper for GARCH. Pricing/mechanism and evaluation papers define what cannot be claimed as new. The proposed distinction is a different, explicit prediction target/information set and measured lead-time comparison; this is a hypothesis about a literature gap, not a certified absence of prior art.

**Claims to avoid:**

- “The first study to predict funding rates,” “a novel RF/GARCH combination,” or “97.5% predictive accuracy.”
- “Model 3 exactly leaks its future target through ma3,” given the verified target shift.
- “The exchange forecast is missing because `predicted_funding_rate` is empty.”
- “There are only repeated settlements in the hourly file” or “42,179 rows equal 42,179 independent trials.”
- “GARCH AIC proves accurate volatility forecasting,” “ADF proves stable regimes,” or “Granger causality proves an economic causal mechanism.”
- “The RF beats persistence,” based on the older lag comparator.
- “Fees are 300 times the alpha,” “no profitable strategy exists,” or “a delta-neutral trade is risk-free.”
- “Generalizes across crypto,” “stress robust,” “unbiased,” “leakage-free” or “state of the art” without the corresponding evidence.
- “March 2024 is an untouched stress holdout,” or “the 2023–2024 period was never used during development.”

**Readiness:** the project currently supports an interesting reproducible audit case, but **does not yet support a defensible completed funding-forecasting paper**. A corrected short technical note may be feasible after clean artifact reproduction and careful positioning. For a substantive preprint/workshop study, complete the minimum experiments and show a nontrivial, well-bounded result. For a stronger journal submission, add untouched-time and preferably external-asset replication or a well-specified uncertainty/decision contribution. These are judgments about evidential strength, not predictions of acceptance.

The highest-value first task is **constructing and validating the event/origin/settlement panel**. It resolves the meaning of the target and determines whether the proposed contribution is feasible. Adding models before that would spend effort without addressing the main scientific uncertainty.

## Reproduction and source notes

Run from the repository root:

```bash
v2/.venv/bin/python v2/research/reproduce_evidence.py
```

The script writes only to this research directory and checks that its three input hashes remain unchanged. It verifies arithmetic and source alignment, not statistical significance or a fresh model fit. The original v1/v2 notebooks and implementation are preserved.

Primary documentation used for label planning: [Binance funding-history endpoint](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History). Historical rule verification is still required; current documentation alone is not evidence that every mechanism parameter was constant from 2020 through 2024.

Search/access date: 25 September 2026. Primary paper links appear next to their summaries. Secondary discovery pages and practitioner examples were not used as evidence for academic findings. Full-text access limitations, unresolved citations and publication-version differences above are part of the novelty assessment and must be resolved before a categorical priority claim.

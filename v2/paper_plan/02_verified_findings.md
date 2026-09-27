# Verified findings and interpretation

### Materials and verification scope

Reviewed all four notebooks, their stored text outputs, preprocessing and model utilities, ingestion code, v2 audit helpers/configuration/tests, README files, the normalized data, and the saved prediction CSVs. Saved pickle files were inventoried but not executed. The original report linked through Canvas is not locally present; its contents were not available. Claims attributed to absent `notes/` files in the existing audit cannot be independently verified from those files.

The locally supplied data contain **42,179 rows and 11 columns**, Binance BTCUSDT only. Nonmissing message-arrival timestamps span **2020-01-01 00:59:57 UTC through 2024-10-17 23:59:59 UTC**. There are **5,256 distinct upcoming funding-event timestamps**. These are event identifiers, not 5,256 independently verified settled rates. The dataset is an hourly-style reduction of derivative-ticker messages, not a transaction tape, order book, or authoritative settlement ledger.

The new [evidence script](../research/reproduce_evidence.py) rereads the input files, verifies row/target alignment, recalculates the headline scores, records input SHA-256 hashes, and creates [machine-readable measurements](evidence/evidence.json), a [score table](evidence/saved_prediction_scores.csv), and a [diagnostic figure](evidence/baseline_comparison.png). It does not rerun the original model training or claim to establish clean out-of-sample performance. Input hashes were unchanged after execution.

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


## Additional evidence collected for implementation planning

The feasibility probe found 5,256 eligible event IDs at all four planned horizons under a 65-minute staleness limit. This counts feature availability under the recorded-arrival assumption, not validated settlements or independent samples. A stricter 60-minute limit reduces each horizon's coverage. See [the full data report](05_data_contract_and_feasibility.md).

Three checksum-verified Binance monthly archives contain 276 funding records. Of these, 275 match a pre-event ticker indication in the local dataset. Across those matches, 47 terminal indications differ from the archive rate by more than 0.01 bp. This is an observed reason to acquire actual labels. It is not evidence of predictability or a representative population error estimate: the three months were deliberately chosen for feasibility and stress coverage.

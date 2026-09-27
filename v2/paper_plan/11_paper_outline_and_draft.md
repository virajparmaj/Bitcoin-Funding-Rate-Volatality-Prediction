# Paper outline and writing material

Working title: **Beyond Persistence: Bitcoin Funding Forecasts Against Exchange Indications at Fixed Settlement Lead Times**.

One-sentence intended contribution: We measure whether simple models add information beyond the exchange's available indication of an upcoming funding payment, and how incremental errors vary with lead time and observable state.

## Manuscript structure

| Section | What to write / evidence needed |
|---|---|
| Abstract | Target, current-indication baseline, dates, exact effect and interval; complete only after E1–E4 |
| Introduction | Why advance notice and incremental information matter; use the saved baseline reversal as motivation |
| Related work | Direct funding forecasts, pricing mechanisms, availability audits; explicit comparison with Inan and newer audit work |
| Questions / hypotheses | Four-hour ridge comparison, secondary horizons/RF, state effect and falsifying outcomes |
| Data | Provider semantics, recorded arrivals, sampled/full settlement reconciliation, provenance limitations and coverage |
| Methodology | Define F, q, residual u, origin and age; explain causal joining and label maturity |
| Experimental setup | Dates, retrospective status, monthly fitting, grids, baselines, matched events, loss and bootstrap choices |
| Results | Main paired effect first; horizon curve second; include unsuccessful models and intervals |
| Robustness / ablations | Delay, feature groups, state/time slices, true extremes, windows/seeds where run |
| Discussion | Which information remains after the public indication; no unsupported causal mechanism or profit claim |
| Limitations | One contract, coarse aggregation, retrospective choices, few extremes, unavailable closest full text if unresolved |
| Conclusion | Answer the declared question with the measured effect or bound; one specific next replication |

## Abstract that can be supported today

Funding forecasts must distinguish updates to a public indication from the rate ultimately settled. We examine an existing Binance BTCUSDT research pipeline containing 42,179 ticker-derived observations. On 7,797 saved next-observation predictions, Random Forest achieves an R² of 0.975 but has 25.6% higher squared error and 49.0% higher absolute error than current-rate persistence. A feature identity reconstructs the current observation rather than the future target, correcting an earlier leakage interpretation. A feasibility audit finds same-event observations at one, two, four and six hours before 5,256 scheduled funding events under a declared staleness limit. Three checksum-verified official settlement archives show that terminal ticker indications cannot be treated as exact settlement labels. These findings motivate an event-aligned comparison of simple residual models against the contemporaneous exchange indication. Incremental settlement forecasting skill remains unmeasured.

This is a diagnostic/planning abstract, not the finished empirical-paper abstract. Replace the final sentences with actual clean forecast estimates and uncertainty after the experiments; do not submit proposed experiments as completed results.

## Argument for the completed paper

Previous work establishes structured funding dynamics and some model-based predictability. The proposed study asks a narrower information question at explicit times before payment. Using verified settlement labels and available ticker indications, compare residual forecasts on identical events, quantify uncertainty and examine predeclared state/time sensitivity. The result should identify either a stable incremental benefit, a bounded absence of benefit for the tested procedures, or an honest unresolved range. The contribution is that measured characterization and its reproducible evidence.

## Writing rules

Keep the historical artifact diagnosis separate from the clean forecast results. Use “basis points” for errors and identify the denominator for every percentage. Call R² a regression score, not accuracy. Cite original sources near claims; explain abstract-only access. Include sample counts beside all conditional results. Keep planned/future-tense sentences out of the finished Results section.

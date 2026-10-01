# Literature positioning and a defensible modest contribution

> Historical design/preparation record. Current execution status, corrections and
> evidence are maintained in [the hardening record](16_hardening_execution.md)
> and [the revised report](../results/settlement_study_v2_1/REPORT.md).

## Updated novelty decision

**Promising, but unproven:** a reproducible multi-year measurement of incremental settlement forecast skill beyond the contemporaneous exchange indication, including lead-time and observable-state dependence. A bounded negative result could be useful if its intervals exclude a declared meaningful gain and its comparison with prior work is explicit.

The following are established ideas: funding forecasting, persistence baselines, leakage prevention, settlement timing, time-varying predictability and funding-arbitrage costs. None should be claimed as this project's conceptual invention.

## New checks in this follow-up

**Zeng, Yang, Han and He (2026), “Point-in-Time Audit Before Alpha.”** This preprint examines Binance BTCUSDT public archives, availability auditing, factor-search budgets and negative economic results over August 2024–August 2026. Its evaluated forecast horizons are 15/60 minutes and its study concerns factor mining and trading evaluation. The inspected methods do not give our proposed current-indication-versus-realized-funding comparison. It substantially weakens “a BTC point-in-time audit” as a standalone novelty claim. Treat it as methodological precedent, not proof of our data quality. [Full text](https://arxiv.org/html/2608.25348v1).

**Siebly practitioner guide, “Funding-Rate Forecasting for Crypto Perpetuals.”** The guide explicitly discusses intraperiod snapshots, realized labels and separate lead-time evaluations, including four hours. It is not evidence of a validated academic forecasting result, but it means the broad proposed workflow is already public. Our contribution must be a measured empirical finding rather than the workflow alone. [Guide](https://siebly.io/research/crypto-funding-rate-forecasting).

The closest academic comparator remains Inan. Its official conference abstract reports double-autoregressive funding forecasts on Binance/Bybit outperforming no-change, with time-varying predictability. SSRN full text remains inaccessible in this session. A different result from our proposed study would not automatically contradict it: targets, origins, periods and baselines may differ. Before claiming distinctness, record its exact timing, input set, comparator and split.

## Contribution comparison

| Proposed angle | Existing overlap | Defensible remaining work |
|---|---|---|
| Forecast Bitcoin funding | Directly overlaps Inan | No novelty claim on the broad task |
| Use GARCH or RF | Established modeling choices | Treat models as instruments of comparison |
| Prevent future-data contamination | Forecast-evaluation literature and Zeng et al. | Necessary validity work; supporting artifact |
| Evaluate fixed settlement lead times | Practitioner guidance; academic overlap still uncertain | Measure a multi-year matched-event skill curve with uncertainty |
| Compare with the current exchange indication | Natural strong benchmark; exact closest-paper coverage unresolved | Quantify residual information beyond that benchmark |
| Analyze +1 bp versus other observed states | Funding mechanics already established | Test a predeclared interaction in incremental errors |
| Find no useful improvement | Negative results already exist in adjacent targets | Bound a specific funding-forecast gain under an explicit information set |

## Full-text novelty gate

Complete a comparison sheet for Inan and any newly found direct competitor: forecast target, origin/lead time, exchange-indication availability, sample, state analysis, uncertainty, code/data access. If all central dimensions overlap, reposition as a replication with a documented robustness extension; do not rename an existing study as new. If overlap is unresolved, use “we evaluate” and “we provide evidence,” not “first.”

Searches on 25 September 2026 included funding forecast lead time, contemporaneous estimate, settlement residual, Inan's title, Binance funding archives and point-in-time BTC audits. The previous search's Google Scholar/Semantic Scholar access limitations still apply. This is a targeted review, not an exhaustive systematic search.

## Earlier detailed literature review

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

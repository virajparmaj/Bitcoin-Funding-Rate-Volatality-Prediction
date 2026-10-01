# Research-to-paper development package

**Implementation update — 1 October 2026:** the development prompt is [15_implementation_brief.md](15_implementation_brief.md). The executable study and measured status are documented in [v2 README](../README.md) and [the execution report](../results/settlement_study_v2_1/REPORT.md). See [the hardening execution record](16_hardening_execution.md) for requirement coverage. The remainder of this package preserves the original planning snapshot.

Prepared 25 September 2026. Status: **research plan plus verified diagnostic results; the proposed settlement forecasting study has not been run**.

The best modest contribution is a multi-year empirical answer to this question: **when, and in which observable market states, can simple models improve on the exchange's current indication of the next funding payment?** The paper should measure incremental information at explicit settlement lead times. It should not claim a new algorithm or the first funding forecast.

Additional searching found close work on BTC availability audits and public guidance already discussing lead-time comparisons. Novelty therefore rests on a defensible empirical result, its uncertainty and reproducible measurement, not merely combining those ideas. Full-text comparison with the closest forecasting paper remains open.

## Read in this order

| File | Purpose |
|---|---|
| [01 Project and scope](01_project_and_scope.md) | What exists, what was investigated, what to reuse |
| [02 Verified findings](02_verified_findings.md) | Repository evidence and corrections to previous interpretations |
| [03 Literature and novelty](03_literature_and_novelty.md) | Competitors, new search findings, defensible contribution |
| [04 Questions and hypotheses](04_questions_and_hypotheses.md) | Precise estimands and falsifiable hypotheses |
| [05 Data contract and feasibility](05_data_contract_and_feasibility.md) | Field definitions, actual feasibility results, remaining data gates |
| [06 Experimental protocol](06_experimental_protocol.md) | Splits, baselines, metrics, inference and experiments |
| [07 Implementation plan](07_implementation_plan.md) | Modules, interfaces, dependencies and completion criteria |
| [08 Test and validation plan](08_test_and_validation_plan.md) | Executed checks versus scientific tests still to implement |
| [09 Results ledger](09_results_ledger.md) | Measured results and explicitly unrun experiment slots |
| [10 Conclusions and claim limits](10_conclusions_and_claim_limits.md) | What can be concluded now; conditional future conclusions |
| [11 Paper outline and draft](11_paper_outline_and_draft.md) | Section-by-section manuscript plan and current-evidence abstract |
| [12 Figures and reproducibility](12_figures_and_reproducibility.md) | Publication displays, artifact schema and rerun instructions |
| [13 Execution backlog](13_execution_backlog.md) | Ordered work packages, effort estimates and decision gates |
| [14 Decision log](14_decision_log.md) | Revisions to the earlier plan and unresolved choices |
| [Evidence index](evidence/README.md) | Scripts, verified source samples, measurements and run manifest |

## Recommendation

Implement E0–E3 first: settlement/timing validation, the four-hour benchmark, matched lead-time comparisons, and information/availability ablations. Add E4's state/time breakdown from the same forecasts to strengthen the modest novelty claim with little extra compute. Preserve extreme observations and report both favorable and unfavorable results. Defer neural networks, elaborate stacking and a trading simulator.

A small additional sensitivity, E3b, scores the same frozen predictions against actual settlements and terminal-indication proxies. It tests whether the observed label discrepancies change model rankings, without training another model.

The implementation plan is complete. Only read-only diagnostics and data-feasibility probes were executed during preparation; new settlement models were not fitted. All new material is inside `v2/paper_plan/`; existing project code and original inputs are preserved. The [earlier full assessment](../research/research_assessment.md) remains supporting background. This package's explicit protocol and decision log govern future implementation where the earlier plan differs.

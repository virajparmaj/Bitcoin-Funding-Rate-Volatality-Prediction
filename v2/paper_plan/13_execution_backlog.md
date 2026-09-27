# Ordered implementation backlog

The estimates below are planning judgments for focused development time, not elapsed-time promises. Existing package/environment setup helps, but unresolved data provenance or full-text access can extend the schedule.

| Order | Work package | Depends on | Acceptance | Rough effort |
|---|---|---|---|---|
| P0 | Resolve closest-paper details and freeze comparison sheet | Source access | Document target/origin/baseline overlap; narrow claim if needed | 0.5–1 day, access dependent |
| P1 | Full labels, checksums and event reconciliation | Public archive probe already passed | Complete event/exclusion manifest and historical timing rules | 1–2 days |
| P2 | Origin panel, availability lineage and causal baselines | P1 | E0 validation and hand-checked baseline predictions | 1–2 days |
| P3 | Features, monthly splits, ridge/RF and protocol freeze | P2 | No temporal-boundary violations; validation-only selection | 1–2 days |
| P4 | E1/E2 results and paired inference | P3 | Main table/horizon plot with uncertainty and coverage | 0.5–1 day |
| P5 | E3/E4 ablations and state/time evidence | P4 | Declared sensitivities, all comparisons retained | 0.5–1 day |
| P6 | Manuscript tables, conclusion, independent rerun | P0/P5 | No unresolved result placeholders or unsupported claims | 1–2 days |

A reasonable initial budget is roughly **6–11 focused development days**, assuming labels and historical availability can be resolved without new feature acquisition. Compute should be measured in the first small fold; no runtime guarantee is implied. Do not pay for additional datasets before P1/P2 establish a concrete need.

## Definition of minimum completed study

- [ ] Full-period official settlement labels and exact source manifest.
- [ ] Causal event-origin panel with exclusion reasons and explicit availability assumptions.
- [ ] Primary four-hour ridge comparison and secondary RF/baselines.
- [ ] Matched lead-time comparison at 1/2/4/6 hours.
- [ ] Controlled feature and one-hour delay ablations.
- [ ] Paired uncertainty, comparison ledger and valid bounded-negative interpretation.
- [ ] Observable-state/time analysis if these appear in the contribution claim.
- [ ] Closest-literature comparison resolves or discloses remaining overlap.
- [ ] Clean rerun reproduces tables, figures and counts.
- [ ] Claims distinguish measured effects from hypotheses and economic speculation.

## Decisions at failure points

If label mapping fails, investigate provenance before fitting anything. If fields cannot be safely timed, omit/delay them and narrow the information set. If the evidence is too imprecise, add a justified untouched sample rather than more hyperparameter trials. If a closest study already covers the same question, pursue a transparent replication/robustness extension. If clean results are null but tightly bounded, write the negative result. If model improvement is stable, report it without inventing a causal or profitability conclusion.

## Next concrete task

Implement P1/P2: cached full-period settlement ingestion, separate scheduled/calculation/availability times, the event-origin panel, and its adversarial tests. This is the highest-value next step and the dependency for every publishable model comparison.

# Implementation prompt: execute the settlement research study

Use this as the project development prompt. Read the current execution report before changing the protocol or rerunning experiments.

## Objective

Convert this repository's proposed BTCUSDT funding study into a reproducible empirical research project. Measure whether simple models improve on the current exchange indication of the upcoming funding settlement, across 1/2/4/6-hour origins and observable states. The contribution is a measured, bounded empirical result; do not invent algorithmic novelty, publication priority or profitability.

## Authoritative inputs

Read `v2/paper_plan/03_literature_and_novelty.md`, `04_questions_and_hypotheses.md`, `05_data_contract_and_feasibility.md`, `06_experimental_protocol.md`, `07_implementation_plan.md`, `08_test_and_validation_plan.md` and the current `v2/results/settlement_study/REPORT.md`. The executable config is `v2/configs/settlement_study.json`. Log justified deviations in the decision log; do not follow superseded target-leakage claims from the original v2 build prompt.

## Required changes

1. Keep original notebooks, utilities, models, source data and saved v1 results unchanged. All research code/output belongs under `v2/`.
2. Acquire public official funding archives with checksum verification, immutable caching and exact source manifests. Preserve calculation time separately from scheduled event and assumed label availability. Reject unexpected schedule, duplicate or ambiguous matches.
3. Build a same-event backward-only origin panel from recorded arrivals. No future-nearest selection, synthetic clocks, outcome imputation or arbitrary duplicate removal. Save exclusions and source-row lineage. Retain true extremes.
4. Use current indication, last known settlement, validation-selected EWMA, training median and a labeled +1 bp reference. Fit residual ridge/RF and a settled-history ridge with training-only preprocessing.
5. Select hyperparameters using monthly 2022 validation only. Freeze the selection before scoring 2023–October 2024; refit monthly on matured labels. Exclude decisions preceding the fit cutoff and keep event horizons together. Call this retrospective evaluation.
6. Execute E1/E2 on matched events, E3 feature/availability ablations, E3b proxy-label rescoring of identical forecasts and E4 descriptive state/time analysis. Save each prediction and every fold before aggregating.
7. Report MAE gain in bp, RMSE, bias, baseline-relative skill, coverage and paired calendar-block intervals. Correct the declared secondary comparison family. Distinguish pointwise intervals from simultaneous bounds and nonsignificance from equivalence.
8. Test future perturbations, origin boundaries, label maturity, checksums, units, missing labels, duplicate events, training-only transforms, calendar gaps and deterministic runs. Tests must exercise scientific failure cases, not just source strings.
9. Generate tables/figures and conclusions from saved outputs. Update documentation and demo status without presenting old next-row metrics as settled-forecast results. Record unexecuted robustness work explicitly.
10. Commit to `viraj/research`, push and open a reviewable pull request. Include executed tests, measured results, protocol deviations, reproduction commands and remaining limitations. Never merge without a separate request.

## Completion criteria

A reviewer can reproduce data validation, model selection, saved forecasts and reported tables from the config and source manifests. The primary four-hour result and uncertainty are stated faithfully even if models lose. The PR preserves original evidence and identifies remaining provenance/novelty limitations. Do not silently tune against test outcomes to obtain a favorable conclusion.

## Run contract

From repository root with the existing v2 environment:

```bash
v2/.venv/bin/python -m v2.experiments.run_study --stage acquire
v2/.venv/bin/python -m v2.experiments.run_study --stage validate-data
v2/.venv/bin/python -m v2.experiments.run_study --stage baselines
v2/.venv/bin/python -m v2.experiments.run_study --stage models
v2/.venv/bin/python -m v2.experiments.run_study --stage report
v2/.venv/bin/python -m pytest -q -c v2/pyproject.toml v2/tests
```

The acquisition stage uses public network reads. Other stages are offline once the archives are cached. Source/implementation changes invalidate cached model selection; create a versioned output directory for a revised protocol instead of overwriting a prior study.

The implementation also provides `v2/.venv/bin/python -m v2.experiments.check_sensitivities` for publication-delay invariance and explicit monthly-boundary exclusion records. Run it before generating the report. If it finds different features or training membership, rerun affected models under separately versioned protocols rather than assuming invariant predictions.

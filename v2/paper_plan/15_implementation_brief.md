# Implementation prompt: execute the settlement research study

Use this as the project development prompt. Read the current execution report before changing the protocol or rerunning experiments.

## Objective

Convert this repository's proposed BTCUSDT funding study into a reproducible empirical research project. Measure whether simple models improve on the current exchange indication of the upcoming funding settlement, across 1/2/4/6-hour origins and observable states. The contribution is a measured, bounded empirical result; do not invent algorithmic novelty, publication priority or profitability.

## Authoritative inputs

Read `v2/paper_plan/03_literature_and_novelty.md`, `04_questions_and_hypotheses.md`, `05_data_contract_and_feasibility.md`, `06_experimental_protocol.md`, `07_implementation_plan.md`, `08_test_and_validation_plan.md` and the current `v2/results/settlement_study_v2_1/REPORT.md`. The executable config is `v2/configs/settlement_study_v2_1.json`. Log justified deviations in the decision log; do not follow superseded target-leakage claims from the original v2 build prompt.

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
study_config=v2/configs/settlement_study_v2_1.json
v2/.venv/bin/python -m v2.experiments.run_study --config "$study_config" --stage acquire
v2/.venv/bin/python -m v2.experiments.run_study --config "$study_config" --stage validate-data
v2/.venv/bin/python -m v2.experiments.run_study --config "$study_config" --stage baselines
v2/.venv/bin/python -m v2.experiments.run_study --config "$study_config" --stage models
v2/.venv/bin/python -m v2.experiments.check_sensitivities --config "$study_config"
v2/.venv/bin/python -m v2.experiments.verify_saved_forecasts --config "$study_config"
v2/.venv/bin/python -m v2.experiments.run_study --config "$study_config" --stage report
v2/.venv/bin/python -m pytest -q -c v2/pyproject.toml v2/tests
```

The acquisition stage uses public network reads only when cached archives are
missing. Other stages are offline. Preserve both previous study directories.
Changes to source, implementation, config or environment invalidate execution
identity: create a new versioned output directory, never overwrite a prior study.

Run the same stages with `v2/configs/settlement_study_v2_1_reproduction.json` for an
independent full execution. Then run `v2.experiments.compare_study_runs` with
`--config` pointing to the revised study and `--other-config` to its reproduction.
The [README](../README.md) explains fresh-checkout behavior and ignored artifacts.

Publication-delay checks must compare exact feature values and both validation
and evaluation membership. Different features or membership require separately
versioned sensitivity fits; do not report invariant forecasts from approximate
equality. Reporting requires matching sensitivity and replay evidence.

A completion pass must preserve the primary comparison and fixed grids, use the
same monthly validation cohort for EWMA and fitted models, verify every completed
stage before writes, publish new stages atomically, and test interruption and
changed-input failures. Update existing PR #1 on `viraj/research`; do not create a
duplicate PR or merge without a separate request.

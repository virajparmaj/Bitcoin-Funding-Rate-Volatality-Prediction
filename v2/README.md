# v2 — Reproducible funding-settlement research

This additive research layer preserves the original coursework and evaluates whether simple models improve on the exchange's current indication of the next BTCUSDT funding settlement. It distinguishes next-row prediction from prediction of an actual payment at a specified lead time.

The implementation follows the [development prompt](paper_plan/15_implementation_brief.md) and [study protocol](configs/settlement_study_v2_1.json). See the [executed report](results/settlement_study_v2_1/REPORT.md) for measured results, uncertainty and outstanding limitations. The original [paper plan](paper_plan/README.md) is the dated design record; it is not an execution log.

## Study

- Checksum-verified Binance public settlement archives, January 2020–October 2024.
- Same-event observations available at 1/2/4/6 hours before settlement; explicit exclusions and age limits.
- Current indication, last known settlement, EWMA, median and +1 bp references.
- Residual ridge and Random Forest, plus a settled-history ridge comparator.
- Monthly 2022 validation for model selection; expanding monthly retrospective evaluation in 2023–October 2024.
- Matched-event lead-time comparison, feature and one-hour availability-delay ablations, proxy-label sensitivity and descriptive state/time slices.
- Paired calendar-block uncertainty and a declared secondary comparison family.

This is an empirical evaluation, not a claim of a new algorithm, first publication, trading profitability or an untouched prospective holdout. Field-level provenance of the original hourly reduction and historical publication timing remain assumptions. Full-text prior-art verification and external/fresh-time replication remain outstanding.

## Run

Use the existing per-project environment or install `v2/requirements.txt` in a Python 3.11 virtual environment. Run from the repository root:

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

Acquisition verifies cached public ZIP/checksum pairs or downloads missing ones;
no credentials are needed. Later stages are offline. The December 2019 warm-up
archive was unavailable; incomplete histories are explicitly excluded.

All commands accept `--config`. Defaults retain the historical config for CLI
compatibility, but the revised implementation refuses to modify that legacy run.
Use the explicit revised config above. Completed stages verify and return without
rewriting. A code, config, source or numerical environment change requires a new
output directory; failed fits never silently substitute a baseline.

For a **fresh checkout**, the derived panels/forecasts are absent by design. Run
the same commands with `v2/configs/settlement_study_v2_1_reproduction.json`, whose
ignored output directory starts empty. To repeat again, copy that JSON to a new
config and change only `output_dir` to another unused child of `v2/results/`.
Do not delete or overwrite a completed run to force execution.

To check full independent reproduction when both runs are present locally:

```bash
v2/.venv/bin/python -m v2.experiments.compare_study_runs \
  --config v2/configs/settlement_study_v2_1.json \
  --other-config v2/configs/settlement_study_v2_1_reproduction.json
```

This compares fresh source-derived panels, eligibility, validation choices,
folds, every prediction and numeric tables. Event membership and parameters must
match exactly; floating outputs use `rtol=1e-9`, `atol=1e-12`. It is same-machine
reproducibility, not independent market replication. The full grid and repeated
forest fitting take substantially longer than the demo.

## Artifacts and release boundaries

The revised run is under `v2/results/settlement_study_v2_1/`. Each successful stage
atomically publishes a directory under `_stages/`, including a receipt of input
and output hashes. Top-level symlinks provide convenient access to the artifacts.
An interrupted publication can repair missing links after verifying canonical
files; partial computations never receive a completion receipt.

The execution identity includes every Python module under `v2/src` and
`v2/experiments`, configuration, ticker, exact archive/checksum hashes and package
versions. Git commit and dirty state are also recorded as metadata. Results
committed later do not change the code used for the recorded run.

The [original September 30 report](results/settlement_study/REPORT.md) remains
unchanged. The [hardening record](paper_plan/16_hardening_execution.md) maps
requirements to evidence and documents remaining limitations.

Local `panel.csv.gz`, `predictions.csv.gz` and baseline forecasts retain per-event
lineage but remain Git-ignored pending redistribution review. Raw archives are
cached under ignored `v2/data/`; manifests contain official URLs and checksums.
Reproduction requires access to the original input CSV and the recorded package
environment. Existing inclusion of the input does not establish redistribution
rights; this is not advertised as an independently accessible open dataset.

## Historical audit corrections

`src/audit.py`, `src/legacy.py` and `notebooks/00_audit_v1.ipynb` reproduce historical artifacts; they are not the clean modeling pipeline. The corrected [assessment](research/research_assessment.md) explains why:

- `3 × ma3 − lag1 − lag2` reconstructs the observable current indication, not Model 3's shifted future target.
- Current-rate persistence beats the saved RF even though both have high next-row R².
- A null following-event estimate does not mean the upcoming-event indication is missing.
- Constant saved test columns do not establish their training contribution.
- AIC magnitude, ADF rejection and fee/MAE ratios do not establish forecast validity or profit.

Original v1 notebooks, utilities, models, source data and saved results stay unchanged. The clean study imports no legacy preprocessing.

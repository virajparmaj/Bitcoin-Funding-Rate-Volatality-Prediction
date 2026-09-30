# v2 — Reproducible funding-settlement research

This additive research layer preserves the original coursework and evaluates whether simple models improve on the exchange's current indication of the next BTCUSDT funding settlement. It distinguishes next-row prediction from prediction of an actual payment at a specified lead time.

The implementation follows the [development prompt](paper_plan/15_implementation_brief.md) and [study protocol](configs/settlement_study.json). See the [executed report](results/settlement_study/REPORT.md) for measured results, uncertainty and outstanding limitations. The original [paper plan](paper_plan/README.md) is the dated design record; it is not an execution log.

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
v2/.venv/bin/python -m v2.experiments.run_study --stage acquire
v2/.venv/bin/python -m v2.experiments.run_study --stage validate-data
v2/.venv/bin/python -m v2.experiments.run_study --stage baselines
v2/.venv/bin/python -m v2.experiments.run_study --stage models
v2/.venv/bin/python -m v2.experiments.check_sensitivities
v2/.venv/bin/python -m v2.experiments.run_study --stage report
v2/.venv/bin/python -m pytest -q -c v2/pyproject.toml v2/tests
```

Acquisition downloads small public label files without credentials. All later stages run offline. The optional December 2019 warm-up archive was unavailable; initial incomplete histories are explicitly excluded. The default full grid fits hundreds of small forests during validation; expect minutes rather than an instantaneous demo.

`--config` selects another JSON protocol. Use a new output directory for a changed protocol; cached selection checks configuration, source and training-code hashes. The forecast stage never substitutes a baseline after a failed fit.

## Artifacts and release boundaries

`v2/results/settlement_study/` contains the executed report, aggregate scores, fold counts, exclusions, source manifest, validation scores, selected parameters and a lead-time figure. Local `panel.csv.gz` and `predictions.csv.gz` retain per-event lineage and forecasts. They are ignored by Git pending redistribution review of provider-derived information. Raw downloaded archives are cached under ignored `v2/data/`; public URLs and checksums allow reacquisition.

The aggregate release is therefore reproducible only with access to the original input file. Do not advertise it as an independently accessible open dataset until source redistribution rights are resolved.

## Historical audit corrections

`src/audit.py`, `src/legacy.py` and `notebooks/00_audit_v1.ipynb` reproduce historical artifacts; they are not the clean modeling pipeline. The corrected [assessment](research/research_assessment.md) explains why:

- `3 × ma3 − lag1 − lag2` reconstructs the observable current indication, not Model 3's shifted future target.
- Current-rate persistence beats the saved RF even though both have high next-row R².
- A null following-event estimate does not mean the upcoming-event indication is missing.
- Constant saved test columns do not establish their training contribution.
- AIC magnitude, ADF rejection and fee/MAE ratios do not establish forecast validity or profit.

Original v1 notebooks, utilities, models, source data and saved results stay unchanged. The clean study imports no legacy preprocessing.

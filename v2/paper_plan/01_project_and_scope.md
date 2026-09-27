# Project reconstruction and reuse strategy

## Intended scientific problem

Predict an upcoming Bitcoin perpetual funding payment from information actually available before settlement. The original implementation mixed three targets: next-row funding changes, contemporaneous funding levels and modeled conditional variance. These are separate tasks; none is automatically evidence of settlement forecasting or trading profit.

## Existing assets

| Asset | Present status | Reuse decision |
|---|---|---|
| `data/normalized_datasets/binance_btc_perp.csv` | 42,179 rows, 11 columns, BTCUSDT ticker reduction | Read-only feature source; preserve hash and missingness |
| `notebooks/stat429_analysis_a.ipynb` | Regression with contemporaneous target and shuffled splitting | Historical diagnostic case only |
| `notebooks/model_development.ipynb` | Classification, GARCH-family fits, RF and SARIMAX experiments | Reuse concepts; clean implementation must replace invalid timing/preprocessing |
| `dgieser3/research.ipynb` | Data exploration and synthetic clock construction | Provenance evidence, not authoritative event time |
| `results/predictions_RFR.csv` | 7,797 saved predictions and features | Reproducible baseline-reversal case |
| `results/predictions_SARIMAX.csv` | 7,622 saved predictions, scale inferred from notebook | Diagnostic only until source execution/units are reproduced |
| `v2/src/audit.py`, `legacy.py`, `config.py` | Implemented audit helpers and configuration | Keep legacy logic isolated; correct scientific interpretation in new work |
| `v2/tests/test_protected_paths.py` | 14 passing checks in this review | Protect evidence paths; does not validate forecasting |
| `models/model2.py`, `model3.py` | Empty | No reusable model implementation here |
| `v2/research/` | Previous detailed assessment and reproducible measurements | Baseline evidence, linked throughout this package |

No authoritative full-period settlement table, clean event-origin panel, out-of-sample residual models or forecast significance results currently exist in this package. Three small official archive samples now establish a feasible route to labels.

## Scope that minimizes work

Use one contract, one venue, the existing date span, one primary horizon and two simple residual models. Add public funding labels instead of repurchasing the full ticker history. Prefer a compact Python experiment runner; notebooks should render results, not conceal stateful training logic. Do not import `utilities` into the new core: its package initialization pulls optional dependencies and the preprocessing preserves known defects.

Funding-state analysis should describe observable conditions, such as whether the current indication equals +1 bp. It should not invent regimes from future outcomes and then imply those regimes were recognizable when forecasting.

## Code evidence anchors

- `utilities/data_processing.py:75`: backward fill before splitting.
- `utilities/data_processing.py:108`: future funding target uses `shift(-1)`.
- `utilities/functions.py:64`: whole-frame zero fill following cleaning.
- `utilities/functions.py:252`: outlier routine to inspect when reproducing legacy behavior.
- `utilities/data_import.py:93`: hourly `resample(...).last()`; columns may select different source messages.
- `utilities/functions.py:169`: current Model 2 helper signature differs from a notebook call.

These are repository-root-relative paths. Keep the original files unchanged; proposed implementation lives under `v2/`.

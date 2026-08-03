# v2 — Bitcoin funding rate: corrected research build

An additive rebuild of this repository's funding-rate work. `v2/` audits the original
submission, reproduces its numbers from source, identifies where the methodology fails,
and rebuilds the study so that the results mean what they claim to mean.

The original submission is **not modified**. It is the evidence being audited, and it
stays byte-for-byte intact so that every claim in the audit can be independently
re-checked against it.

---

## The question

Binance's BTCUSDT perpetual settles a funding payment every 8 hours. The research
question is whether the next settlement's funding rate can be forecast well enough, and
early enough, to be worth acting on after transaction costs.

That last clause is the whole problem. Funding rates are extremely persistent, so a
model can post a very high R² while contributing nothing over "assume it stays the
same." Any honest answer has to be stated as an improvement over a named baseline,
denominated in basis points, net of fees.

---

## What the audit found

`notebooks/00_audit_v1.ipynb` reproduces 51 of 55 measured claims about the v1 build
directly from the raw data and the committed result CSVs. The four exceptions are
recorded as discrepancies in the notebook rather than quietly reconciled.

The headline findings:

| # | Finding | Evidence |
|---|---|---|
| 1 | Model 3's design matrix contains a closed-form reconstruction of its own target, `fr(t) = 3·ma3 − lag1 − lag2`. The closed form scores **0.9801**, beating the trained model's **0.9750** | §7 |
| 2 | SARIMAX is reported at R² = 0.925; recomputed from the CSV v1 itself wrote, it is **−4.807**. `.predict(start, end)` past the sample is a 7,622-step forecast, scored as if one-step | §8 |
| 3 | Analysis A's 0.936 uses a contemporaneous target, a rolling window containing that target, and a shuffled split on a series with lag-1 autocorrelation 0.9887 | §6 |
| 4 | No baseline was ever computed. Naive persistence alone scores **0.937** on the same test set | §7 |
| 5 | The `timestamp` column is synthetic — a manufactured hourly grid that drifts **130 hours** from the exchange clock, matching the true hour-of-day only 9.2% of the time | §2 |
| 6 | `bfill()` on the full frame fills **8,531 cells** from future observations before any split | §3 |
| 7 | Outlier removal plus whole-frame `fillna(0)` rewrites **1,118 real funding observations** averaging 12.44 bp as exactly zero | §4 |
| 8 | Three "model output" columns in the shipped test set are constant, so Models 1 and 2 contribute nothing to Model 3 | §9 |
| 9 | The measured edge over persistence is 0.0303 bp against a ~9 bp round trip — **~300× too small** | §10 |
| 10 | The GARCH selection AIC of −732,756 is a units artefact; the identical fit scores +44,210 under a different scaling | §11 |

The common thread is finding 4. Every other defect stayed invisible because no metric
was ever reported next to a baseline. That is the ordering v2 inverts.

---

## How v2 is built

**Baselines before models.** Phase 2 produces a full results table — persistence, EWMA,
HAR-RV, majority class, and the exchange's own published forecast — before any v2 model
exists. No metric is ever reported without its baseline in the same table.

**Point-in-time correctness is enforced, not assumed.** `src/validate.py` asserts that
no feature is constant, that no feature set permits reconstructing the target (the
`3·ma3 − lag1 − lag2` path is tested by name), that no backward fill appears anywhere,
and that timestamps never become features. The tests fail on the v1 feature set and
pass on the v2 one.

**The sample is allowed to look worse.** The real UTC clock is rebuilt from
`local_timestamp`; dropped days stay as explicit NaN gaps with an `is_gap` flag rather
than being interpolated. Extreme funding settlements are real events and are kept.

**Evaluation is purged and embargoed.** Walk-forward with an embargo of at least one
settlement, Diebold-Mariano against persistence with Newey-West HAC lags = 8, and
metrics bucketed by funding-volatility regime with the March 2024 spike held out as a
stress sample.

**Everything lands in basis points, net of costs.** If the conclusion is that no trade
clears the fee, that is the finding, stated plainly.

---

## Layout

```
v2/
├── README.md                    this file
├── requirements.txt             UTF-8, pinned to v1's core versions
├── src/
│   ├── config.py                paths, fees, windows, seed — no value is hardcoded elsewhere
│   ├── legacy.py                faithful reimplementation of v1 logic — AUDIT ONLY
│   ├── audit.py                 measurement helpers for the audit (read-only)
│   ├── timebase.py              real UTC clock rebuilt from local_timestamp        [Phase 1]
│   ├── settlement.py            hourly rows -> per-settlement panel                [Phase 1]
│   ├── baselines.py             persistence, EWMA, HAR-RV, majority, exchange      [Phase 2]
│   ├── metrics.py               bp-denominated, Diebold-Mariano (HAC), cost-aware  [Phase 2]
│   ├── features.py              point-in-time safe: no fillna(0), no pct_change    [Phase 3]
│   ├── validate.py              leakage and constant-feature assertions            [Phase 3]
│   └── splits.py                purged + embargoed walk-forward                    [Phase 4]
├── notebooks/                   00..07, thin — one or two calls into src per cell
├── tests/                       pytest, targeting the numerical and feature code
├── data/                        derived only, gitignored
└── results/                     v2 outputs only — never ../results/
```

### Why `legacy.py` exists

`v2/src/` cannot import `utilities/`. `utilities/__init__.py` star-imports `data_pull`
at package-init time, which imports `tardis_dev`, and `utilities/functions.py` imports
`imblearn` at module scope — neither is in v1's `requirements.txt`:

```
from utilities.data_processing import process_pipeline
-> ModuleNotFoundError: No module named 'imblearn'
```

Patching `utilities/` to make it importable would modify the artefact under audit, so
`legacy.py` reimplements the v1 code paths faithfully instead — defects included, with
the originating `file:line` cited in each docstring. It is used only by the audit.

---

## Running it

```bash
python3.11 -m venv v2/.venv
v2/.venv/bin/pip install -r v2/requirements.txt
```

Then, from `v2/notebooks/`:

```bash
../.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace 00_audit_v1.ipynb
```

The core scientific stack is pinned to the same versions as v1's `requirements.txt`
(numpy 2.1.2, pandas 2.2.3, scikit-learn 1.5.2, statsmodels 0.14.4, scipy 1.14.1) so
that reproductions are not confounded by library drift. v1's file is UTF-16 encoded and
omits `arch`, `tardis_dev`, and `imbalanced-learn` despite importing all three; the v2
file is UTF-8 and includes them.

Phase 0 runs entirely offline against files already in the repository. No API key is
needed.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Scaffolding, `config.py`, `legacy.py`, `00_audit_v1.ipynb` | **Complete** |
| 1 | `timebase.py`, `settlement.py`, notebooks 01–02 | Not started |
| 2 | `baselines.py`, `metrics.py`, notebook 03 | Not started |
| 3 | `features.py`, `validate.py`, notebook 04, tests | Not started |
| 4 | `splits.py`, notebook 05 | Not started |
| 5 | Economics and findings, notebooks 06–07 | Not started |
| 6 | Tardis re-pull of `predicted_funding_rate`, cross-exchange dispersion | **Blocked** — needs `TARDIS_API_KEY` and explicit sign-off before any network call |

### Known limitations

- `predicted_funding_rate` is 100% null in the shipped file, so the most informative
  formulation of the problem — forecasting the *residual* against the exchange's own
  published estimate — is unavailable until Phase 6.
- The data is a single instrument on a single venue. Cross-exchange funding dispersion
  (OKX, Bybit, Hyperliquid) is the obvious next predictor and is also Phase 6.
- The audit's Analysis A correction variants reproduce in sign and magnitude but not to
  the quoted decimal; see the discrepancy note in §12 of the audit notebook.

---

## Protected paths

v2 never writes to any of these. `config.assert_writable()` raises on any attempt:

```
notebooks/  dgieser3/  utilities/  models/  results/  data/  config.py
test_codebase.py  requirements.txt  README.md
```

"""Export research definitions and existing notebook evidence without executing research.

Run with Python 3 from any directory. Uses only the standard library; never imports
project modules, loads serialized models, trains models, or writes research inputs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "demo/public/data/research-content.json"
SOURCES: list[dict] = []
READ_PATHS: set[str] = set()


def read(path: str) -> str:
    """Read an evidence input and remember its path for provenance."""
    READ_PATHS.add(path)
    raw = (ROOT / path).read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    return raw.decode(encoding)


def lines(source_id: str, path: str, start: int, end: int, kind="Executable definition"):
    """Capture exact one-based source lines for the local evidence viewer."""
    excerpt = "\n".join(read(path).splitlines()[start - 1 : end])
    SOURCES.append(dict(id=source_id, path=path, locator=f"lines {start}–{end}",
                        excerpt=excerpt, evidenceType=kind))
    return source_id


def cell(source_id: str, path: str, index: int, output: int | None = None):
    """Capture a zero-based notebook JSON cell or a stored text output."""
    notebook_cell = json.loads(read(path))["cells"][index]
    if output is None:
        excerpt = "".join(notebook_cell["source"])
        kind = "Reported in project notes" if notebook_cell["cell_type"] == "markdown" else "Executable definition"
        locator = f"JSON cell {index} ({notebook_cell['cell_type']})"
    else:
        stored = notebook_cell["outputs"][output]
        excerpt = stored.get("text") or stored.get("data", {}).get("text/plain")
        if excerpt is None:
            excerpt = f"{stored.get('ename', '')}: {stored.get('evalue', '')}"
        excerpt = "".join(excerpt) if isinstance(excerpt, list) else excerpt
        kind = "Stored execution output"
        locator = f"JSON cell {index}, output {output}, execution_count={notebook_cell.get('execution_count')}"
    SOURCES.append(dict(id=source_id, path=path, locator=locator, excerpt=excerpt, evidenceType=kind))
    return source_id


def section(source_id: str, path: str, heading: str, kind="Reported in project notes"):
    """Capture a markdown section with exact line locations."""
    content = read(path).splitlines()
    start = next(i for i, line in enumerate(content) if line == heading)
    end = next((i for i in range(start + 1, len(content)) if content[i].startswith("## ")), len(content))
    return lines(source_id, path, start + 1, end, kind)


ANALYSIS = "notebooks/stat429_analysis_a.ipynb"
MODELS = "notebooks/model_development.ipynb"
cell("analysis-preprocessed", ANALYSIS, 1, 0)
cell("analysis-features", ANALYSIS, 22)
cell("analysis-split", ANALYSIS, 25)
for name, index in [("lr1", 27), ("lr2", 28), ("rf", 30)]:
    cell(f"analysis-{name}-code", ANALYSIS, index)
    cell(f"analysis-{name}-output", ANALYSIS, index, 2)
cell("analysis-adf-code", ANALYSIS, 13)
cell("analysis-adf-output", ANALYSIS, 13, 0)
cell("analysis-arima-code", ANALYSIS, 32)
cell("analysis-arima-warning", ANALYSIS, 32, 0)
cell("analysis-arima-output", ANALYSIS, 32, 1)
cell("analysis-conflicting-summary", ANALYSIS, 33)
for name, index in [("pipeline", 1), ("classifier-rf", 5), ("classifier-logistic", 7),
                    ("classifier-reported", 9), ("garch", 11), ("garch-reported", 12),
                    ("regression-rf", 14), ("regression-reported", 15), ("regression-sarimax", 17)]:
    cell(name, MODELS, index)
cell("pipeline-stored-error", MODELS, 1, 1)
cell("synthetic-clock", "dgieser3/research.ipynb", 2)
lines("preprocess", "utilities/data_processing.py", 47, 79)
lines("target-calendar", "utilities/data_processing.py", 85, 117)
lines("pipeline-sequence", "utilities/data_processing.py", 177, 209)
lines("lag-definitions", "utilities/functions.py", 49, 64)
lines("technical-definitions", "utilities/functions.py", 85, 99)
lines("interaction-definitions", "utilities/functions.py", 121, 131)
lines("direction-integration", "utilities/functions.py", 139, 167)
lines("variance-integration", "utilities/functions.py", 169, 190)
lines("outlier-definition", "utilities/functions.py", 252, 258)
lines("tuning", "utilities/model_utils.py", 171, 193)
lines("standalone-model1", "models/model1.py", 21, 78)
lines("config-input", "config.py", 18, 45)
lines("import-sampling", "utilities/data_import.py", 83, 110)
lines("group-attribution", "README.md", 1, 10, "Reported in project notes")
section("notes-classification", "notes/09_results_and_metrics.md", "## Model 1 Direction Prediction")
section("notes-analysis", "notes/09_results_and_metrics.md", "## Analysis A Baselines")
section("notes-garch", "notes/09_results_and_metrics.md", "## Model 2 Volatility Prediction")
section("notes-regression", "notes/09_results_and_metrics.md", "## Model 3 Exact Funding Rate")
lines("notes-features", "notes/05_feature_engineering.md", 11, 43, "Reported in project notes")
lines("notes-models", "notes/06_ml_models.md", 11, 78, "Reported in project notes")
lines("assessment-stale-context", "v2/research/research_assessment.md", 11, 15, "Reported in project notes")
lines("paper-stale-context", "v2/paper_plan/02_verified_findings.md", 1, 7, "Reported in project notes")
section("v2-readme-status", "v2/README.md", "## Status")
section("v2-readme-claims", "v2/README.md", "## What the audit found")
lines("v2-planned-modules", "v2/paper_plan/07_implementation_plan.md", 1, 27, "Planned")
section("v2-pending-results", "v2/paper_plan/09_results_ledger.md", "## New model experiments", "Planned")
section("v2-feasibility", "v2/paper_plan/05_data_contract_and_feasibility.md", "## Official label feasibility probe")
section("v2-backlog", "v2/paper_plan/13_execution_backlog.md", "## Definition of minimum completed study", "Planned")
cell("old-audit-identity", "v2/notebooks/00_audit_v1.ipynb", 28)
cell("old-audit-economics", "v2/notebooks/00_audit_v1.ipynb", 38)

saved_headers = {}
for label, path in [("Model 3 · RF", "results/predictions_RFR.csv"), ("Model 3 · SARIMAX", "results/predictions_SARIMAX.csv")]:
    saved_headers[label] = next(csv.reader(read(path).splitlines()))
    lines("saved-header-rf" if label.endswith("RF") else "saved-header-sarimax", path, 1, 1, "Saved artifact")

src_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "v2/src").glob("*.py"))
for path in src_files:
    read(path)
SOURCES.append(dict(id="v2-source-inventory", path="v2/src/", locator="Python source-file inventory",
                    excerpt="\n".join(src_files), evidenceType="Current repository inspection"))
model_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "models").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
SOURCES.append(dict(id="model-inventory", path="models/", locator="Current file inventory; no serialized models loaded",
                    excerpt="\n".join(model_files), evidenceType="Current repository inspection"))

TASKS = [
    dict(id="analysis-a", title="Exploratory analysis", subtitle="Contemporaneous funding-rate regression",
         target="funding_rate[t] — the current observed funding indication", units="Native funding rate; MSE in rate²; R² unitless",
         inputs=["Open interest and mark price", "Calendar variables from local_timestamp", "24-row funding standard deviation in LR2 and RF"],
         models=["Linear Regression 1", "Linear Regression 2", "Random Forest Regressor", "ARIMA(5,1,0) full-series fit"],
         split="Regression: shuffled 80/20 train_test_split, random_state=42. ARIMA: fit to the available full series.",
         preprocessing=["Preprocessed data; stored initial output shows missing values filled.", "The regression design matrix drops missing feature rows.", "Time series, histogram, correlations, rolling summaries, ADF, ACF and PACF are included."],
         limitations=["Predicts the current response; this is not verified future forecasting.", "The 24-row std contains the current response in LR2 and RF.", "Shuffling ignores chronological separation.", "Printed regression 'next five' values reuse the last five test inputs.", "Stored outputs reflect historical execution; current helpers and saved execution history differ."],
         status="Stored execution output", sourceIds=["analysis-features", "analysis-split", "analysis-lr1-output", "analysis-lr2-output", "analysis-rf-output", "analysis-arima-code"]),
    dict(id="model-1", title="Model 1 · Direction", subtitle="Next-observation classification",
         target="direction[t] = 1 if funding_rate[t+1] > funding_rate[t], otherwise 0", units="Class 1: increase; class 0: unchanged or decrease",
         inputs=["Funding lags, rolling means and EMA3", "Mark price and its lag", "RF also selects open-interest history, rate-of-change and interactions"],
         models=["Logistic Regression", "Random Forest Classifier"],
         split="Chronological 80/20 holdout in notebook code; ordinary five-fold GridSearchCV for tuning.",
         preprocessing=["Feature generation runs before the holdout split.", "Notebook RF setup enables funding rescaling by 10^6 and outlier handling; logistic cell calls pipeline defaults.", "StandardScaler is fitted on training data before GridSearchCV; preprocessing is not refitted inside each CV fold.", "Nonfinite RF features are replaced with zero; logistic preprocessing forward/backfills."],
         limitations=["No saved classification predictions or usable execution metrics were found in the inspected training cells.", "Reported RF accuracy and F1 conflict between notebook narrative and project notes.", "Ordinary CV is not a demonstrated walk-forward evaluation.", "The current target helper assigns the final unknown next-row comparison to class 0. The standalone model1.py path separately drops its missing future target.", "volatility_5min is requested while the current helper creates volatility_5h."],
         status="Implemented, result unavailable", sourceIds=["classifier-rf", "classifier-logistic", "classifier-reported", "notes-classification", "tuning", "target-calendar", "standalone-model1"]),
    dict(id="model-2", title="Model 2 · Conditional variance", subtitle="GARCH-family fitting",
         target="Conditional variance of the funding-rate series; terminal forecast at five observation steps", units="Conditional variance in squared rate units; standard deviation is its square root",
         inputs=["funding_rate series from process_pipeline defaults", "Autoregressive mean terms for AR(1) and AR(2) variants"],
         models=["GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH variants", "Normal, Student-t and GED innovation alternatives"],
         split="Full-series fitting and in-sample AIC/BIC comparison; no verified out-of-sample variance evaluation.",
         preprocessing=["Current default pipeline forward/backfills and constructs features.", "ADF code conditionally differences the series.", "Selects the minimum-AIC fitted model and prints terminal conditional variances; plot takes their square roots."],
         limitations=["GJR-GARCH(1,1) with AR(2) mean is selected in narrative only; the fitting cell has no stored output.", "AIC/BIC compare in-sample fit with complexity penalties; they do not measure future variance accuracy.", "The fitting code uses funding_rate, without a Model 1 direction input.", "Absolute parameter magnitudes are not feature importance.", "Narrative variance values are not independently verified."],
         status="Implemented, result unavailable", sourceIds=["garch", "garch-reported", "notes-garch", "variance-integration"]),
    dict(id="model-3", title="Model 3 · Funding level", subtitle="Next-observation funding-rate regression",
         target="future_funding_rate[t] = funding_rate[t+1]", units="RF saved values: native funding rate; native rate × 10,000 = bp. SARIMAX path rescales funding by 10^6.",
         inputs=["Funding lags, smoothers, changes, interactions and calendar features", "Market and time fields selected by exclusion", "Intended direction and conditional-variance outputs"],
         models=["Random Forest Regressor", "SARIMAX(3,1,3) with exogenous features"],
         split="Chronological 80/20 split in source. RF and SARIMAX saved outputs cover different samples; SARIMAX predicts the holdout in one block.",
         preprocessing=["Target shift and missing-row removal precede the split.", "RF replaces nonfinite values with zero and clips features to ±10^9.", "SARIMAX rescales funding by 10^6, enables outlier handling, and clips features to ±10^5.", "Current helper interfaces disagree with notebook integration calls."],
         limitations=["Saved predictions establish artifact-level scores, not a successful current end-to-end training run.", "Constant saved integration columns do not establish a contribution during training.", "The current-rate identity reconstructs an observable predictor, not the unknown next-row target.", "Retrospective persistence diagnostics are later comparisons, not original fitted models.", "SARIMAX's block forecast, units and sample differ from the RF comparison."],
         status="Recomputed from saved predictions", sourceIds=["regression-rf", "regression-sarimax", "direction-integration", "variance-integration", "pipeline-stored-error", "saved-header-rf", "saved-header-sarimax"]),
]

# Extract exact existing printed metrics; these numbers are never fitted here.
METRICS = []
for short, title in [("lr1", "Linear Regression 1"), ("lr2", "Linear Regression 2"), ("rf", "Random Forest Regressor")]:
    text = next(s["excerpt"] for s in SOURCES if s["id"] == f"analysis-{short}-output")
    values = {"r2": float(re.search(r"R²: ([^\n]+)", text).group(1)),
              "mse": float(re.search(r"MSE: ([^\n]+)", text).group(1))}
    METRICS.append(dict(id=f"analysis-{short}", taskId="analysis-a", model=title,
                        status="Stored execution output", target="Current funding_rate[t]",
                        units="R²: unitless; MSE: native funding-rate²", sampleCount=None,
                        sampleCountNote="Test count is not printed in the stored metric cell. The stored initial frame has 42,179 rows; 24-row rolling std and the 80/20 split imply 8,432 test rows if executed sequentially as shown. This count was not independently reproduced.",
                        evaluation="Shuffled 80/20 split; random_state=42; historical notebook output, not a fresh run.",
                        values=values, sourceIds=[f"analysis-{short}-output", f"analysis-{short}-code", "analysis-features", "analysis-split", "analysis-preprocessed"]))

RF_SELECTION = {"funding_rate_lag1", "funding_rate_lag2", "funding_rate_ma3", "funding_rate_ma5", "funding_rate_ema3", "open_interest", "open_interest_lag1", "open_interest_roc", "mark_price", "mark_price_lag1", "volatility_5min", "funding_rate_roc1", "funding_rate_roc3", "interaction2", "interaction3"}
LR_SELECTION = {"funding_rate_lag1", "funding_rate_lag2", "funding_rate_ma5", "funding_rate_ema3", "mark_price", "mark_price_lag1", "funding_rate_ma3"}
ANALYSIS_SELECTIONS = {"Analysis A · LR1": {"open_interest", "mark_price", "hour", "day", "month"}, "Analysis A · LR2": {"open_interest", "mark_price", "std", "day", "month"}, "Analysis A · RF": {"open_interest", "mark_price", "hour", "day", "month", "std"}}
FEATURES = []


def feature(name, group, formula, purpose, engineered_in, refs, limitations=()):
    """Keep definitions, source selection and saved-column presence separate."""
    selected = [label for label, names in ANALYSIS_SELECTIONS.items() if name in names]
    if name in RF_SELECTION:
        selected.append("Model 1 · RF")
    if name in LR_SELECTION:
        selected.append("Model 1 · Logistic")
    saved = [label for label, header in saved_headers.items() if name in header]
    selected.extend(saved)  # Model 3 source selects every feature by exclusion.
    source_ids = list(refs)
    if "Model 1 · RF" in selected:
        source_ids.append("classifier-rf")
    if "Model 1 · Logistic" in selected:
        source_ids.append("classifier-logistic")
    if any(s.startswith("Analysis") for s in selected):
        source_ids.extend(["analysis-features", "analysis-lr1-code", "analysis-lr2-code"])
    if saved:
        source_ids.extend(["regression-rf", "regression-sarimax", "saved-header-rf", "saved-header-sarimax"])
    FEATURES.append(dict(id=name, name=name, group=group, formula=formula, purpose=purpose,
                         purposeStatus="Design rationale; not a measured contribution", engineeredIn=engineered_in,
                         selectedBy=selected, savedIn=saved, limitations=list(limitations), sourceIds=list(dict.fromkeys(source_ids))))

for lag in (1, 2):
    feature(f"funding_rate_lag{lag}", "Funding history", f"funding_rate.shift({lag})", "Represent recent funding history.", "utilities/functions.py · add_lag_features", ["lag-definitions"], ["Lookback is in rows, not independently verified hours. Lag creation fills all missing cells with zero."])
for window in (3, 5):
    feature(f"funding_rate_ma{window}", "Smoothed funding", f"funding_rate.rolling({window}).mean()", "Summarize the current and recent funding observations.", "utilities/functions.py · add_technical_indicators", ["technical-definitions"], ["Includes the current funding indication; not inherently a future-target leak for a next-row target."])
    feature(f"funding_rate_ema{window}", "Smoothed funding", f"funding_rate.ewm(span={window}, adjust=False).mean()", "Smooth history with more weight on recent observations.", "utilities/functions.py · add_technical_indicators", ["technical-definitions"])
for lag in (1, 3):
    feature(f"funding_rate_roc{lag}", "Funding changes", f"funding_rate.pct_change(periods={lag})", "Represent proportional funding changes.", "utilities/functions.py · add_technical_indicators", ["technical-definitions"], ["Zero or near-zero denominators can create infinities or unstable ratios; later zero-filling/clipping changes their meaning."])
for name in ("open_interest", "mark_price", "last_price", "index_price"):
    feature(name, "Market state", "Observed ticker field; selected and transformed by experiment", "Provide market-state context.", "Raw input field", ["config-input", "import-sampling"], ["Hourly resample .last() can select last nonmissing fields asynchronously; exact original message provenance is unavailable."])
for name in ("open_interest", "mark_price"):
    feature(f"{name}_lag1", "Market state", f"{name}.shift(1)", "Represent the preceding market-state observation.", "utilities/functions.py · add_lag_features", ["lag-definitions"])
feature("open_interest_roc", "Market state", "open_interest.pct_change(periods=1)", "Represent proportional open-interest changes.", "utilities/functions.py · add_technical_indicators", ["technical-definitions"], ["Zero-denominator and missing-data cleanup apply."])
feature("volatility_5h", "Price variability", "mark_price.rolling(window=5).std()", "Summarize dispersion of five mark-price levels.", "utilities/functions.py · add_technical_indicators", ["technical-definitions", "regression-rf", "regression-sarimax"], ["Standard deviation of price levels, not returns or funding rates. Window means five rows; helper name does not verify a regular hourly clock.", "This name is absent from saved prediction headers; legacy paths request volatility_5min. Model 3's all-column selector would include the current helper column if execution reached that stage; current end-to-end execution is unverified."])
feature("volatility_5min", "Price variability", "Legacy feature name; current helper emits volatility_5h instead", "Legacy price-variability input selected by notebook/integration.", "Historical implementation not recovered; present in notebook selectors and saved outputs", ["technical-definitions", "direction-integration"], ["The name is not evidence of five-minute data. Do not silently equate historical implementation with the current helper."])
feature("std", "Funding variability", "funding_rate.rolling(window=24).std()", "Summarize 24 funding observations including the current response.", "notebooks/stat429_analysis_a.ipynb · JSON cell 22", ["analysis-features"], ["Contains funding_rate[t], the contemporaneous target of Analysis A. No measured forecasting contribution follows."])
for name, formula, purpose in [
    ("interaction1", "funding_rate_lag1 × funding_rate_lag2", "Represent a nonlinear relation between recent funding levels."),
    ("interaction2", "MA3 / (lag1.replace(0, NaN) + 1e-6); infinities → NaN → 0", "Represent relative smoothed-to-lagged funding, with explicit cleanup."),
    ("interaction3", "mark_price_lag1 × funding_rate_ma3", "Represent a funding/price cross-term."),
]:
    feature(name, "Interactions", formula, purpose, "utilities/functions.py · add_interaction_terms", ["interaction-definitions"], ["Purpose is a design rationale; no causal or ablation contribution was measured."])
for part, period in [("hour", 24), ("day", 31), ("month", 12)]:
    feature(part, "Calendar", f"timestamp.dt.{part}" + (" (day of month)" if part == "day" else ""), "Represent calendar position.", "utilities/data_processing.py · create_features; Analysis A uses local_timestamp", ["target-calendar", "analysis-features", "synthetic-clock"], ["Model-development calendar uses synthetic timestamp; Analysis A extracts from local_timestamp."])
    for trig in ("sin", "cos"):
        feature(f"{part}_{trig}", "Calendar", f"{trig}(2π × {part} / {period})", "Encode cyclical calendar position.", "utilities/data_processing.py · create_features", ["target-calendar", "synthetic-clock"], ["Synthetic time provenance applies; day is day-of-month with fixed period 31."])
feature("model1_direction_pred", "Intended model outputs", "loaded Model 1 RF.predict(scaler.transform(selected features))", "Intended integration of direction into Model 3.", "utilities/functions.py · add_model1_direction", ["direction-integration"], ["Constant in saved RF test rows; this does not establish its effects during training. No verified integration gain."])
feature("model2_volatility_h1", "Intended model outputs", "model2_result.forecast(reindex=False).variance.iloc[:,0], reindexed to df", "Intended integration of conditional variance into Model 3.", "utilities/functions.py · add_model2_volatility", ["variance-integration"], ["Variance, not standard deviation. Constant in saved RF test rows; no verified integration gain.", "Notebook calls steps=5, but the current helper requires model2_result."])

# Small saved-column checks are descriptive arithmetic, never model execution.
rf_rows = list(csv.DictReader(read("results/predictions_RFR.csv").splitlines()))
constant_checks = {}
for name in ("model1_direction_pred", "model2_volatility_h1"):
    unique_values = sorted({float(row[name]) for row in rf_rows if row[name] != ""})
    constant_checks[name] = {"rowCount": len(rf_rows), "uniqueValueCount": len(unique_values), "values": unique_values}
SOURCES.append(dict(id="saved-integration-values", path="results/predictions_RFR.csv",
                    locator="Descriptive scan of model1_direction_pred and model2_volatility_h1 over every saved row",
                    excerpt=json.dumps(constant_checks, indent=2), evidenceType="Recomputed from saved predictions"))
for item in FEATURES:
    if item["name"] in constant_checks:
        item["sourceIds"].append("saved-integration-values")
TASKS[3]["sourceIds"].append("saved-integration-values")

FINDINGS = [
    dict(id="separate-targets", title="One dataset, distinct questions", text="Exploration predicts the current indication; classifiers predict a next-row increase; GARCH fits conditional variance; Model 3 predicts the next observed rate. Their scores answer different questions.", status="Confirmed from executable definitions", sourceIds=["analysis-features", "target-calendar", "garch", "regression-rf"]),
    dict(id="feature-work", title="Real feature engineering, explicit scope", text="The implementation creates lags, smoothers, proportional changes, interactions and cyclical calendar encodings. Experiment selections differ, and design rationales are not measured feature contributions.", status="Confirmed from executable definitions", sourceIds=["lag-definitions", "technical-definitions", "interaction-definitions", "target-calendar", "classifier-rf", "classifier-logistic"]),
    dict(id="evaluation-design", title="Evaluation design changes the claim", text="Analysis A combines a shuffled split with a current-rate target and a target-containing variability feature. Its stored regression scores are exploratory evidence, not verified future forecasts.", status="Stored execution output", sourceIds=["analysis-features", "analysis-split", "analysis-rf-output"]),
    dict(id="integration", title="An intended stack, an unresolved contribution", text="Model 3 attempts to add direction and conditional variance. Both columns are constant in its saved RF test data; no verified integration improvement is available, and training effects cannot be inferred from test constants.", status="Recomputed from saved predictions", sourceIds=["regression-rf", "saved-integration-values"]),
    dict(id="reproducibility", title="Saved artifacts need separate interpretation", text="Notebook outputs, current helper interfaces and narrative results disagree. The later audit adds useful saved-result checks, while settlement forecasting and clean walk-forward model comparisons remain planned.", status="Current repository inspection", sourceIds=["pipeline-stored-error", "variance-integration", "notes-classification", "v2-pending-results", "v2-source-inventory"]),
]

LIMITATIONS = [
    dict(id="splits", title="Splits and tuning", text="Analysis A shuffles the holdout. Model 1 and Model 3 use chronological holdouts in source, but classifier tuning uses ordinary five-fold GridSearchCV, after train-wide scaling. Neither establishes a clean walk-forward protocol.", sourceIds=["analysis-split", "classifier-rf", "classifier-logistic", "tuning", "regression-rf"]),
    dict(id="preprocessing", title="Future-aware fills and zero replacement", text="Current preprocessing forward-fills then backward-fills the full frame before splitting. The lag helper fills all missing values with zero. In the legacy outlier path, a shortened funding series is reassigned, creating missing cells that later become zero. These changes can alter labels and feature meaning.", sourceIds=["preprocess", "pipeline-sequence", "lag-definitions", "outlier-definition"]),
    dict(id="target-boundary", title="The final direction label", text="The target helper evaluates a missing final future observation as false, then casts it to class 0. It does not itself drop that unknown target. The standalone logistic script has a separate dropna step, so target validity depends on the execution path.", sourceIds=["target-calendar", "standalone-model1"]),
    dict(id="timestamp", title="Time and sampling provenance", text="Ingestion resamples derivative tickers nominally hourly; a later notebook overwrites timestamp with a synthetic hourly grid. Observation lookbacks are row counts, and a synthetic axis must not be relabeled as verified UTC arrival time.", sourceIds=["import-sampling", "synthetic-clock", "config-input"]),
    dict(id="interfaces", title="Current helper and notebook mismatch", text="The current helper creates volatility_5h, but notebook/integration code expects volatility_5min. Model 3 calls add_model2_volatility(df, steps=5), while the current helper requires a fitted model result. Saved predictions are not proof that this checkout trains end to end.", sourceIds=["technical-definitions", "direction-integration", "variance-integration", "regression-rf", "regression-sarimax"]),
    dict(id="stored-error", title="Stored history is not a current rerun", text="The saved setup output reports an unexpected keep_future_rate argument and then a NoneType error. The current create_features signature now accepts that argument. The error documents historical execution state; it does not verify that the same error occurs now.", sourceIds=["pipeline-stored-error", "target-calendar", "pipeline"]),
    dict(id="garch", title="Variance fitting versus forecast accuracy", text="GARCH AIC/BIC are in-sample selection criteria. The reported selected model and terminal variance forecast lack stored execution output here, and no usable out-of-sample variance score is verified. Absolute parameter values should not be shown as feature importance.", sourceIds=["garch", "garch-reported"]),
    dict(id="metric-conflicts", title="Conflicting reported classification scores", text="The notebook and notes report different tuned RF accuracy and F1 values. Keep them in optional historical detail; there is no verified classification leaderboard, ROC curve or confusion matrix to reconstruct from summaries.", sourceIds=["classifier-reported", "notes-classification"]),
    dict(id="economics", title="Prediction errors do not establish profits", text="No verified strategy, executable backtest or trading-profit result is available in the inspected work. An error comparison cannot prove the existence or absence of profitable trading, and the older audit's fee-to-error comparison is not an economic validation.", sourceIds=["old-audit-economics", "v2-pending-results", "v2-backlog"]),
]

PLANS = [
    dict(id="settlement-labels", title="Validate settlement labels", status="Planned", text="Build complete official settlement-label coverage, event reconciliation and an availability-aware origin panel. Existing archive samples are feasibility checks, not completed full-period forecasts.", sourceIds=["v2-feasibility", "v2-planned-modules", "v2-pending-results"]),
    dict(id="lead-time", title="Forecast at defined lead times", status="Planned", text="Compare current indications, settled-history baselines and residual ridge/RF models at fixed lead times on matched events.", sourceIds=["v2-planned-modules", "v2-pending-results"]),
    dict(id="walk-forward", title="Use chronological evaluation", status="Planned", text="Implement expanding monthly splits, causal feature availability, validation-only selection and controlled delay/feature ablations.", sourceIds=["v2-planned-modules", "v2-pending-results"]),
    dict(id="uncertainty", title="Measure uncertainty and robustness", status="Planned", text="Add paired gain intervals, dependence-aware testing, tail/window/seed sensitivity and untouched-time or external validation.", sourceIds=["v2-pending-results", "v2-backlog"]),
    dict(id="economics", title="Evaluate a defined trading policy", status="Planned", text="Only after forecasting validation, test a specified policy with actual cash flows, execution assumptions, turnover and costs. No economic result is claimed here.", sourceIds=["v2-backlog", "v2-readme-status"]),
]

CORRECTIONS = [
    dict(id="notes-present", title="The project notes are present", staleClaim="The newer research assessment and copied verified-findings document describe notes/ as absent.", correction="notes/05_feature_engineering.md, notes/06_ml_models.md and notes/09_results_and_metrics.md exist in this checkout and were read. Their narrative claims still require artifact-level verification.", sourceIds=["assessment-stale-context", "paper-stale-context", "notes-features", "notes-models", "notes-classification"]),
    dict(id="identity", title="Current-rate reconstruction is a baseline", staleClaim="The older v2 README/audit calls 3×MA3−lag1−lag2 a reconstruction of Model 3's target.", correction="The identity reconstructs funding_rate[t]. Model 3 explicitly targets funding_rate[t+1]. It supplies an observable current-rate persistence predictor; the identity alone does not establish future-target leakage.", sourceIds=["v2-readme-claims", "old-audit-identity", "technical-definitions", "lag-definitions", "regression-rf"]),
    dict(id="garch-dependency", title="No implemented direction-to-GARCH edge", staleClaim="Notebook narrative describes direction feeding Model 2.", correction="The GARCH fitting cell extracts funding_rate and passes that series to arch_model. It does not consume Model 1 predictions.", sourceIds=["garch", "direction-integration"]),
    dict(id="analysis-metrics", title="Stored outputs take precedence over summaries", staleClaim="Analysis A markdown and project notes quote different Linear/RF scores and an RF MSE that does not match the stored execution output.", correction="The primary table uses exact metrics parsed from notebook outputs. LR2 retains a stored output despite a null execution_count; no new fitting was performed.", sourceIds=["analysis-lr1-output", "analysis-lr2-output", "analysis-rf-output", "analysis-conflicting-summary", "notes-analysis"]),
    dict(id="v2-implementation", title="The corrected study remains a plan", staleClaim="Some v2 README prose describes baseline, validation and walk-forward modules in the present tense.", correction="The actual Python sources are __init__.py, config.py, legacy.py and audit.py. The paper-plan results ledger marks new experiments pending. Preliminary data-feasibility files do not change that status.", sourceIds=["v2-source-inventory", "v2-readme-status", "v2-planned-modules", "v2-pending-results", "v2-feasibility"]),
]

# Narrative numbers are kept isolated from primary metrics, never inferred as executions.
HISTORICAL = [
    dict(id="classification-narrative", taskId="model-1", title="Conflicting reported classification results", status="Reported in project notes", sampleCount=None,
         text="Notebook narrative reports Logistic Regression accuracy 80.26% and F1 65.89%; tuned RF accuracy 78.75% and F1 71.27%. Notes instead report tuned RF accuracy 77.38%, F1 71.42% and ROC AUC 0.8718. These are conflicting narrative claims, not verified comparable runs.",
         target="Next-observation increase versus unchanged/decrease", units="Accuracy/F1 percentages; ROC AUC unitless", evaluation="Narrative only; exact result-to-run mapping not verified.", sourceIds=["classifier-reported", "notes-classification"]),
    dict(id="garch-narrative", taskId="model-2", title="Reported GJR-GARCH selection", status="Reported in project notes", sampleCount=None,
         text="The notebook narrative selects GJR-GARCH(1,1) with an AR(2) mean and describes rising terminal five-step conditional variance. Its numerical AIC/BIC and forecast claims are not promoted to verified results because the fitting cell contains no stored output.",
         target="Funding-rate conditional variance", units="Variance is squared funding-rate units", evaluation="Reported in-sample model selection; no verified out-of-sample variance score.", sourceIds=["garch", "garch-reported"]),
]

source_ids = {source["id"] for source in SOURCES}
for collection in (TASKS, FEATURES, METRICS, FINDINGS, LIMITATIONS, PLANS, CORRECTIONS, HISTORICAL):
    for item in collection:
        assert set(item["sourceIds"]).issubset(source_ids), item["id"]
for name, result in constant_checks.items():
    assert result["uniqueValueCount"] == 1, f"Saved constant-column claim changed: {name}"

payload = dict(
    schemaVersion=1,
    provenance=dict(method="Static source inspection and extraction of stored notebook text outputs; no experiments or model loading.",
                    notebookLocatorConvention="JSON cell and output indices are zero-based; file line numbers are one-based.",
                    attribution="Repository coursework and later evaluation audit; individual model ownership is not inferred.",
                    inputHashes={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sorted(READ_PATHS)}),
    tasks=TASKS, features=FEATURES, metrics=METRICS, findings=FINDINGS, limitations=LIMITATIONS,
    plans=PLANS, corrections=CORRECTIONS, historical=HISTORICAL, sources=SOURCES,
    artifactInventory=dict(v2Sources=src_files, modelFiles=model_files, integrationColumns=constant_checks),
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
print(f"Exported {len(TASKS)} tasks, {len(FEATURES)} features, {len(METRICS)} stored result rows and {len(SOURCES)} sources to {OUTPUT.relative_to(ROOT)}")

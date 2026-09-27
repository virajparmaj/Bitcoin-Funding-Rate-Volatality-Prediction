#!/usr/bin/env python3
"""Export existing artifacts for the local demo. Python standard library only.

No research code is imported or executed. No model is loaded or trained. All
writes are confined to demo/public/data. Inputs are hashed before and after.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
OUT = DEMO / "public/data"
BP = 10_000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
RECOMPUTED = "Recomputed from saved predictions"
RAW = "data/normalized_datasets/binance_btc_perp.csv"
RF = "results/predictions_RFR.csv"
SAR = "results/predictions_SARIMAX.csv"
SOURCES = [RAW, RF, SAR, "v2/research/evidence.json",
           "v2/research/saved_prediction_scores.csv", "v2/research/reproduce_evidence.py",
           "v2/research/research_assessment.md", "utilities/functions.py",
           "utilities/data_processing.py", "utilities/data_import.py",
           "notebooks/model_development.ipynb", "dgieser3/research.ipynb"]
FIELD_INFO = {
    "exchange": ("Identifiers", "text", "Provider exchange identifier."),
    "symbol": ("Identifiers", "text", "Provider instrument identifier."),
    "local_timestamp": ("Time", "microseconds since Unix epoch", "Message-arrival time; valid values define raw calendar charts."),
    "funding_timestamp": ("Time", "microseconds since Unix epoch", "Upcoming funding-event identifier; not an independently verified settlement label."),
    "funding_rate": ("Funding", "native decimal rate", "Observed ticker funding indication, not an authoritative settled payment."),
    "predicted_funding_rate": ("Funding", "native decimal rate", "Provider field, entirely missing here; not this project's model prediction."),
    "open_interest": ("Market", "provider-native units (unit not verified locally)", "Reported open interest; absolute units are not certified by a local units manifest."),
    "last_price": ("Market", "USDT per BTC", "Last traded price field; selection differs by experiment."),
    "index_price": ("Market", "USDT per BTC", "Underlying index-price field."),
    "mark_price": ("Market", "USDT per BTC", "Mark-price field used by current feature helpers."),
    "timestamp": ("Time", "microseconds since Unix epoch", "Synthetic row grid; unsuitable as verified market UTC."),
}
MISSING = {"", "nan", "NaN", "NA", "null", "None"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes() -> dict[str, str]:
    return {p: sha256(ROOT / p) for p in SOURCES}


def read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def number(value: str) -> float | None:
    if value in MISSING:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric source value: {value!r}")
    return result


def microseconds(value: str) -> int | None:
    return None if value in MISSING else int(Decimal(value))


def utc(value: int) -> str:
    return (EPOCH + timedelta(microseconds=value)).isoformat(timespec="microseconds").replace("+00:00", "Z")


def iso_microseconds(value: str) -> int:
    delta = datetime.fromisoformat(value) - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def score(actual: list[float], predicted: list[float]) -> dict[str, float | int]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Metrics require equally sized, nonempty paired data")
    residuals = [a - p for a, p in zip(actual, predicted)]
    mse = fmean(r * r for r in residuals)
    mean_actual = fmean(actual)
    denominator = fmean((a - mean_actual) ** 2 for a in actual)
    if denominator == 0:
        raise ValueError("R² is undefined for a constant target")
    return {"n": len(actual), "r2": 1 - mse / denominator,
            "mae_native": fmean(abs(r) for r in residuals),
            "rmse_native": math.sqrt(mse), "mse_native": mse,
            "mae_bp": fmean(abs(r) for r in residuals) * BP,
            "rmse_bp": math.sqrt(mse) * BP}


def summarize(values: list[float], row_count: int) -> dict:
    return {"count": len(values), "missing_count": row_count - len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": fmean(values) if values else None}


def daily_series(raw: list[dict[str, str]]) -> list[dict]:
    """Keep every UTC day and true extremes; null means no valid value."""
    grouped = defaultdict(list)
    for row in raw:
        arrival = microseconds(row["local_timestamp"])
        if arrival is not None:
            grouped[utc(arrival)[:10]].append(row)
    cursor = date.fromisoformat(min(grouped))
    end = date.fromisoformat(max(grouped))
    output = []
    while cursor <= end:
        key = cursor.isoformat()
        rows = grouped[key]
        record = {"date": key, "row_count": len(rows), "expected_count": 24,
                  "nominal_shortfall": max(0, 24 - len(rows)),
                  "nominal_excess": max(0, len(rows) - 24),
                  "missing_day": not rows}
        for field in ("funding_rate", "mark_price", "open_interest"):
            values = [number(row[field]) for row in rows]
            record[field] = summarize([v for v in values if v is not None], len(rows))
        record["incomplete_day"] = (len(rows) != 24 or
                                    any(record[field]["missing_count"] for field in
                                        ("funding_rate", "mark_price", "open_interest")))
        output.append(record)
        cursor += timedelta(days=1)
    return output


def notebook_source(path: str, index: int) -> str:
    notebook = json.loads((ROOT / path).read_text())
    return "".join(notebook["cells"][index]["source"])


def source_records(hashes: dict) -> list[dict]:
    source = []
    for id_, path, locator, excerpt in [
        ("raw", RAW, "CSV header and all 42,179 rows", (ROOT / RAW).read_text().splitlines()[0]),
        ("rf", RF, "Actual, Predicted and funding-history columns; all saved rows", (ROOT / RF).read_text().splitlines()[0]),
        ("sarimax", SAR, "Actual and Predicted; all saved rows", (ROOT / SAR).read_text().splitlines()[0]),
        ("rf-target", "notebooks/model_development.ipynb", "JSON cell index 14 (zero based)", notebook_source("notebooks/model_development.ipynb", 14)),
        ("sarimax-setup", "notebooks/model_development.ipynb", "JSON cell index 17 (zero based)", notebook_source("notebooks/model_development.ipynb", 17)),
        ("synthetic-time", "dgieser3/research.ipynb", "JSON cell index 2 (zero based)", notebook_source("dgieser3/research.ipynb", 2)),
        ("feature-helpers", "utilities/functions.py", "add_lag_features / add_technical_indicators / add_interaction_terms", "\n".join((ROOT / "utilities/functions.py").read_text().splitlines()[32:135])),
        ("research-scores", "v2/research/saved_prediction_scores.csv", "Complete saved diagnostic score table", (ROOT / "v2/research/saved_prediction_scores.csv").read_text()),
        ("research-evidence", "v2/research/evidence.json", "Saved artifact-verification measurements", (ROOT / "v2/research/evidence.json").read_text()),
        ("research-reproduction", "v2/research/reproduce_evidence.py", "Read-only research diagnostic (not executed by demo export)", "\n".join((ROOT / "v2/research/reproduce_evidence.py").read_text().splitlines()[34:49])),
    ]:
        source.append({"id": id_, "path": path, "locator": locator, "excerpt": excerpt,
                       "sha256": hashes[path]})
    return source


def build_exports() -> tuple[dict, dict]:
    hashes = input_hashes()
    fields, raw = read_csv(RAW)
    _, rf = read_csv(RF)
    _, sar = read_csv(SAR)
    assert fields == list(FIELD_INFO), "Raw schema changed; review before exporting"
    n = len(raw)
    columns = []
    for name, (group, unit, definition) in FIELD_INFO.items():
        missing = sum(row[name] in MISSING for row in raw)
        columns.append({"name": name, "group": group, "unit": unit,
                        "missing_count": missing, "valid_count": n - missing,
                        "missing_pct": 100 * missing / n, "definition": definition,
                        "source": "raw", "derivation": "Count nonempty source CSV values; no filling or cleaning."})
    arrivals = [microseconds(row["local_timestamp"]) for row in raw]
    valid_arrivals = [x for x in arrivals if x is not None]
    synthetic = [microseconds(row["timestamp"]) for row in raw]
    events = defaultdict(set)
    for row in raw:
        key = microseconds(row["funding_timestamp"])
        if key is not None:
            if number(row["funding_rate"]) is not None:
                events[key].add(number(row["funding_rate"]))
            else:
                events[key]
    adjacent_gaps = [(b - a) / 1_000_000 for a, b in zip(arrivals, arrivals[1:])
                     if a is not None and b is not None]
    preview_indices = list(range(5))
    for field in ("funding_rate", "local_timestamp"):
        index = next((i for i, row in enumerate(raw) if row[field] in MISSING), None)
        if index is not None and index not in preview_indices:
            preview_indices.append(index)
    preview = []
    for i in preview_indices:
        row = raw[i]
        record = {"row_index": i}
        for key, value in row.items():
            record[key] = (None if value in MISSING else value) if key in ("exchange", "symbol") else number(value)
        preview.append(record)

    rf_rows = []
    for i, row in enumerate(rf):
        actual, predicted = number(row["Actual"]), number(row["Predicted"])
        lag1, lag2, ma3 = (number(row[k]) for k in ("funding_rate_lag1", "funding_rate_lag2", "funding_rate_ma3"))
        assert None not in (actual, predicted, lag1, lag2, ma3, number(row["funding_rate_ema3"]))
        rf_rows.append({"row_index": i, "source_row_index": int(row["index"]),
                        "actual": actual, "predicted": predicted,
                        "current": 3 * ma3 - lag1 - lag2, "ema3": number(row["funding_rate_ema3"]),
                        "lag1": lag1, "residual": actual - predicted})
    actual = [row["actual"] for row in rf_rows]
    labels = [("current", "Current-rate persistence"), ("predicted", "Saved Random Forest"),
              ("ema3", "EMA3"), ("lag1", "Older lag1 observation")]
    metrics = []
    for id_, label in labels:
        metric = score(actual, [row[id_] for row in rf_rows])
        metric.update({"id": id_, "label": label, "source": RF,
                       "sources": ["rf", "rf-target", "research-scores"],
                       "task": "Model 3: next-observation funding-rate regression",
                       "target": "funding_rate[t+1], stored as Actual",
                       "units": {"r2": "unitless", "mae_native": "decimal rate", "mse_native": "decimal rate squared", "mae_bp": "basis points", "rmse_bp": "basis points"},
                       "evaluation": "Arithmetic on all saved RF pairs. Later diagnostic comparison; no training or new holdout.",
                       "status": RECOMPUTED})
        metrics.append(metric)
    by_id = {record["id"]: record for record in metrics}
    comparison = {"excess_mse_pct": (by_id["predicted"]["mse_native"] / by_id["current"]["mse_native"] - 1) * 100,
                  "excess_mae_pct": (by_id["predicted"]["mae_native"] / by_id["current"]["mae_native"] - 1) * 100,
                  "skill_vs_current": 1 - by_id["predicted"]["mse_native"] / by_id["current"]["mse_native"],
                  "n": len(rf), "status": RECOMPUTED, "source": RF,
                  "task": "Paired RF / current-rate diagnostic", "target": "next observed funding indication",
                  "units": "percent excess error; unitless squared-error skill",
                  "evaluation": "All 7,797 identical saved RF rows; no retraining."}
    raw_by_key = {microseconds(row["timestamp"]): i for i, row in enumerate(raw)}
    assert len(raw_by_key) == len(raw), "Synthetic row key is not unique"
    aligned_current, aligned_next, linked_arrivals = [], [], []
    for saved, output in zip(rf, rf_rows):
        raw_index = raw_by_key[iso_microseconds(saved["timestamp"])]
        arrival = arrivals[raw_index]
        if arrival is not None:
            linked_arrivals.append(arrival)
        raw_current = number(raw[raw_index]["funding_rate"])
        raw_next = number(raw[raw_index + 1]["funding_rate"]) if raw_index + 1 < len(raw) else None
        if raw_current is not None and raw_next is not None:
            aligned_current.append(abs(output["current"] - raw_current))
            aligned_next.append(abs(output["actual"] - raw_next))
    assert max(aligned_current) < 1e-12 and max(aligned_next) < 1e-12, "Saved target alignment changed"
    alignment = {"total_rows": len(rf), "rows_with_both_raw_values": len(aligned_current),
                 "current_max_abs_discrepancy": max(aligned_current), "next_max_abs_discrepancy": max(aligned_next),
                 "linked_arrival_start_utc": utc(min(linked_arrivals)), "linked_arrival_end_utc": utc(max(linked_arrivals)),
                 "method": "Synthetic timestamp used only as a unique row-identity join key; compare reconstruction with raw current and Actual with raw next. Four rows lack a raw value; no saved prediction is removed from scoring.",
                 "chart_axis": "Zero-based saved row index, never synthetic timestamp labeled as UTC."}
    constants = {}
    for name in ("model1_direction_pred", "model2_volatility_h1", "predicted_funding_rate", "local_timestamp"):
        values = sorted({number(row[name]) for row in rf})
        constants[name] = {"distinct_count": len(values), "values": values}

    sar_score = score([number(row["Actual"]) for row in sar], [number(row["Predicted"]) for row in sar])
    sarimax = {"n": len(sar), "r2": sar_score["r2"],
               "mae_saved_units": sar_score["mae_native"], "rmse_saved_units": sar_score["rmse_native"], "mse_saved_units": sar_score["mse_native"],
               "mae_bp_assuming_scale": sar_score["mae_native"] / 1e6 * BP,
               "rmse_bp_assuming_scale": sar_score["rmse_native"] / 1e6 * BP,
               "assumed_scaling_factor": 1e6,
               "scale_note": "Notebook JSON cell 17 sets rescale=True, scaling_factor=1e6; saved CSV has no units manifest. Conditional bp conversion is saved-unit error / 1,000,000 × 10,000.",
               "source": SAR, "sources": ["sarimax", "sarimax-setup"], "status": RECOMPUTED,
               "task": "Model 3: SARIMAX block prediction diagnostic", "target": "future_funding_rate = funding_rate.shift(-1), with experiment scaling",
               "units": {"r2": "unitless", "mae_saved_units": "saved target units", "rmse_saved_units": "saved target units", "mse_saved_units": "saved target units squared"},
               "evaluation": "All 7,622 saved pairs; test-period exogenous features, block prediction, no sequential target-state updating shown. Sample and setup differ from RF: not a matched comparison."}
    daily = daily_series(raw)
    evidence = {
        "schema_version": 1,
        "purpose": "Local presentation of existing saved artifacts; no fresh training, clean prospective validation, settlement backtest or profitability claim.",
        "input_sha256": hashes,
        "dataset": {"source": RAW, "row_count": n, "column_count": len(fields),
                    "exchanges": sorted({row["exchange"] for row in raw if row["exchange"] not in MISSING}),
                    "symbols": sorted({row["symbol"] for row in raw if row["symbol"] not in MISSING}),
                    "arrival_start_utc": utc(min(valid_arrivals)), "arrival_end_utc": utc(max(valid_arrivals)),
                    "synthetic_start_utc": utc(min(synthetic)), "synthetic_end_utc": utc(max(synthetic)),
                    "funding_event_count": len(events),
                    "events_with_multiple_distinct_rates": sum(len(rates) > 1 for rates in events.values()),
                    "varying_event_share": sum(len(rates) > 1 for rates in events.values()) / len(events),
                    "one_bp_count": sum(number(row["funding_rate"]) == .0001 for row in raw),
                    "one_bp_share_all_rows": sum(number(row["funding_rate"]) == .0001 for row in raw) / n,
                    "arrival_valid_count": len(valid_arrivals),
                    "adjacent_gaps_over_90min": sum(gap > 5400 for gap in adjacent_gaps),
                    "adjacent_negative_gaps": sum(gap < 0 for gap in adjacent_gaps),
                    "synthetic_minus_arrival_first_hours": (synthetic[0] - arrivals[0]) / 3_600_000_000,
                    "synthetic_minus_arrival_last_hours": (synthetic[-1] - arrivals[-1]) / 3_600_000_000,
                    "calendar_day_count": len(daily), "calendar_missing_days": sum(day["missing_day"] for day in daily),
                    "columns": columns, "preview": preview,
                    "caveats": ["Nominally hourly derivative-ticker observations; rows are not independent funding payments.",
                                "Use valid local_timestamp for calendar charts. Synthetic timestamp is a row grid, not verified market UTC.",
                                "Funding-event IDs do not certify realized settlement labels; funding_rate is an evolving ticker indication.",
                                "predicted_funding_rate is wholly unavailable and is unrelated to this project's model predictions.",
                                "Adjacent gap checks omit pairs containing the missing arrival timestamp; near-hour boundaries can create calendar-bin collisions."]},
        "rf": {"n": len(rf), "metrics": metrics, "comparison": comparison, "constant_columns": constants, "alignment": alignment},
        "sarimax": sarimax,
        "sources": source_records(hashes),
        "derivations": {"basis_points": "native decimal rate × 10,000", "current": "3 × funding_rate_ma3 − funding_rate_lag1 − funding_rate_lag2",
                        "residual": "Actual − Predicted (native decimal rate)",
                        "mae": "mean(abs(Actual − prediction))", "mse": "mean((Actual − prediction)^2)",
                        "rmse": "sqrt(MSE)", "r2": "1 − sum((Actual − prediction)^2) / sum((Actual − mean(Actual))^2)",
                        "raw_daily": "UTC day of valid local_timestamp; nonmissing field mean plus exact min/max and valid/missing counts. No interpolation, filling, or outlier removal.",
                        "nominal_reference": "24 is a nominal daily row-count reference only, not proof of missing observations; daily bin counts are affected by arrival-time boundaries.",
                        "rf_axis": "All saved rows retained in CSV order, zero-based. source_row_index is the saved index column, not a verified market timestamp."},
        "publication": {"public_release_cleared": False,
                        "note": "Local demo only. Provider-derived aggregates and saved predictions may carry redistribution restrictions. No redistribution permission is established in this export; review provider rights before public deployment. Full raw CSV is not copied."}}
    series = {"schema_version": 1,
              "units": {"funding_rate": "native decimal rate; multiply by 10,000 for bp", "rf": "native decimal rates; multiply by 10,000 for bp",
                        "mark_price": "USDT per BTC", "open_interest": "provider-native units; not independently verified", "date": "UTC calendar date from valid local_timestamp"},
              "aggregation": evidence["derivations"]["raw_daily"],
              "nominal_reference": evidence["derivations"]["nominal_reference"],
              "excluded_raw_rows_missing_arrival": n - len(valid_arrivals), "rf_axis": evidence["derivations"]["rf_axis"],
              "raw_daily": daily, "rf": rf_rows}
    assert hashes == input_hashes(), "A research input changed during export"
    return evidence, series


def serialized(value: dict, compact: bool = False) -> bytes:
    return (json.dumps(value, indent=None if compact else 2, separators=(",", ":") if compact else None,
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_export(path: Path, payload: bytes) -> None:
    resolved = path.resolve()
    if OUT.resolve() not in resolved.parents:
        raise PermissionError("Exports must remain within demo/public/data")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(payload)


def main() -> None:
    before = input_hashes()
    evidence, series = build_exports()
    outputs = {"evidence.json": serialized(evidence), "series.json": serialized(series, compact=True)}
    for filename, payload in outputs.items():
        write_export(OUT / filename, payload)
    # The sibling exporter owns this file; include, but never rewrite, its output.
    # Refuse a stale content manifest rather than certifying obsolete sources.
    combined_inputs = dict(before)
    research_path = OUT / "research-content.json"
    supplemental_generators = {}
    if research_path.exists():
        research_bytes = research_path.read_bytes()
        research = json.loads(research_bytes)
        for path, expected in research["provenance"]["inputHashes"].items():
            if sha256(ROOT / path) != expected:
                raise ValueError(f"Research content input changed: {path}; rerun export_research_content.py first")
            combined_inputs[path] = expected
        outputs[research_path.name] = research_bytes
        supplemental_generators["research-content.json"] = {
            "generator": "demo/scripts/export_research_content.py",
            "sha256": sha256(DEMO / "scripts/export_research_content.py")}
    manifest = {"schema_version": 1, "generator": "demo/scripts/export_evidence.py",
                "generator_sha256": sha256(Path(__file__)), "command": "demo/.venv/bin/python demo/scripts/export_evidence.py",
                "supplemental_generators": supplemental_generators,
                "deterministic": True, "input_sha256": dict(sorted(combined_inputs.items())),
                "outputs": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in outputs.items()},
                "publication": evidence["publication"], "purpose": evidence["purpose"],
                "evidence_status_definitions": {RECOMPUTED: "Arithmetic reproduced from existing saved pairs; no fresh fit.",
                                                "Stored execution output": "Notebook output present; not rerun.",
                                                "Reported in project notes": "Narrative claim; not independently reproduced.",
                                                "Implemented, result unavailable": "Code exists; usable result unverified.",
                                                "Planned": "Future work; excluded from completed comparisons."}}
    write_export(OUT / "manifest.json", serialized(manifest))
    assert before == input_hashes(), "Source hashes changed during export"
    assert combined_inputs == {path: sha256(ROOT / path) for path in combined_inputs}, "A research input changed during export"
    print(f"Exported {evidence['dataset']['row_count']:,} raw-row summaries, {len(series['raw_daily']):,} UTC daily bands, and {len(series['rf']):,} RF pairs.")
    print("Research inputs unchanged. Local demo exports are not cleared for public redistribution.")


if __name__ == "__main__":
    main()

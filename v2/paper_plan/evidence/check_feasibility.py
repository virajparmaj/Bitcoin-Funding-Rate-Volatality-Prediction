"""Offline planning diagnostics; not a trained-model or held-out forecast experiment.

Reads the original dataset and three checksum-verified public funding archives.
Writes only feasibility.json alongside this file. No API key or network is used.
"""

from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main() -> None:
    source = ROOT / "data/normalized_datasets/binance_btc_perp.csv"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    raw = pd.read_csv(source)
    data = pd.DataFrame({
        "event": pd.to_datetime(raw.funding_timestamp, unit="us", utc=True),
        "arrival": pd.to_datetime(raw.local_timestamp, unit="us", utc=True),
        "rate": raw.funding_rate,
    }).dropna().sort_values("arrival")
    groups = dict(tuple(data.groupby("event")))
    rows = []
    for event, group in groups.items():
        for horizon in (1, 2, 4, 6):
            origin = event - pd.Timedelta(hours=horizon)
            eligible = group[group.arrival <= origin]
            if eligible.empty:
                continue
            last = eligible.iloc[-1]
            age = (origin - last.arrival).total_seconds() / 60
            rows.append({"event": event, "horizon": horizon,
                         "age_minutes": age, "rate": last.rate})
    panel = pd.DataFrame(rows)
    coverage = []
    for horizon in (1, 2, 4, 6):
        sample = panel[panel.horizon == horizon]
        for tolerance in (30, 60, 65):
            selected = sample[sample.age_minutes <= tolerance]
            coverage.append({"horizon_hours": horizon, "max_age_minutes": tolerance,
                             "n": len(selected), "median_age_minutes":
                             float(selected.age_minutes.median())})
    admitted = panel[panel.age_minutes <= 65]
    counts = admitted.groupby("event").horizon.nunique()
    common = counts[counts == 4].index
    years = pd.Series(common.year).value_counts().sort_index().to_dict()

    probes = json.loads((OUT / "archive_probe.json").read_text())
    comparisons = []
    for probe in probes:
        if probe["status"] != "verified":
            continue
        path = OUT / probe["url"].split("/")[-1]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == probe["sha256"]
        with zipfile.ZipFile(path) as archive:
            with archive.open(archive.namelist()[0]) as handle:
                labels = pd.read_csv(handle)
        labels["actual_calc_time"] = pd.to_datetime(labels.calc_time, unit="ms", utc=True)
        # Explicit diagnostic matching only: preserve the original time, permit at
        # most 60 seconds from the nearest hour, then require a unique event key.
        labels["candidate_event"] = labels.actual_calc_time.dt.round("h")
        delta = (labels.actual_calc_time - labels.candidate_event).dt.total_seconds()
        assert (delta.abs() <= 60).all()
        assert labels.candidate_event.is_unique
        assert (labels.candidate_event.dt.hour % 8 == 0).all()
        assert (labels.funding_interval_hours == 8).all()
        differences = []
        last_ages = []
        for row in labels.itertuples():
            if row.candidate_event not in groups:
                continue
            candidates = groups[row.candidate_event]
            candidates = candidates[candidates.arrival < row.candidate_event]
            if candidates.empty:
                continue
            last = candidates.iloc[-1]
            differences.append(abs(last.rate - row.last_funding_rate) * 1e4)
            last_ages.append((row.candidate_event - last.arrival).total_seconds() / 60)
        comparisons.append({
            "month": probe["month"], "archive_rows": len(labels),
            "matched_pre_event_indications": len(differences),
            "calc_time_offset_seconds_min": float(delta.min()),
            "calc_time_offset_seconds_max": float(delta.max()),
            "nonzero_calc_time_offsets": int((delta != 0).sum()),
            "terminal_proxy_mae_bp": float(np.mean(differences)),
            "terminal_proxy_max_abs_error_bp": float(np.max(differences)),
            "terminal_proxy_errors_over_001bp": int((np.array(differences) > .01).sum()),
            "terminal_proxy_median_age_minutes": float(np.median(last_ages)),
        })
    result = {
        "status": "planning feasibility only; no model fit or significance test",
        "input_sha256": digest,
        "usable_event_groups": len(groups), "coverage": coverage,
        "common_events_all_four_horizons_at_65min": len(common),
        "common_events_by_year": {str(k): int(v) for k, v in years.items()},
        "archive_sample_comparisons": comparisons,
        "label_match_rule": "Nearest hour <=60s, unique and 8h-aligned; diagnostic only. Retain calc_time and scheduled event separately in the final implementation.",
        "availability_caveat": "Recorded row arrival is a conservative proxy conditional on valid upstream aggregation; field-level provenance remains unverified.",
    }
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    (OUT / "feasibility.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

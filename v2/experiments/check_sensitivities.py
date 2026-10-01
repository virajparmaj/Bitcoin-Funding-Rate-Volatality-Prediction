"""Exact offline publication-delay and event-boundary diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v2.experiments.run_study import DEFAULT_CONFIG, panel_for, run, save_json
from v2.src.splits import monthly_folds
from v2.src.study_store import digest


def membership(panel, start, end):
    """Include every event/horizon training and evaluation key."""
    return [
        (cutoff, set(zip(train.event_id, train.horizon)), set(zip(test.event_id, test.horizon)))
        for cutoff, train, test in monthly_folds(panel, start, end)
    ]


def compare_panels(reference, candidate, columns, spec):
    """Exact keyed equality; even tiny genuine feature changes require refitting."""
    keys = ["event_id", "horizon"]
    a, b = (p.set_index(keys).sort_index() for p in (reference, candidate))
    validation_end = (pd.Timestamp(spec["test_start"]) - pd.Timedelta(nanoseconds=1)).isoformat()
    ranges = [(spec["validation_start"], validation_end), (spec["test_start"], spec["test_end"])]
    return dict(
        same_origin_keys=a.index.equals(b.index) and a.origin.equals(b.origin),
        identical_settled_features=a[columns].equals(b[columns]),
        identical_fold_membership=all(
            membership(reference, start, end) == membership(candidate, start, end)
            for start, end in ranges
        ),
    )


def check(output, spec, ticker, labels, identity):
    baseline, _ = panel_for(ticker, labels, spec)
    columns = ["settled_1", "settled_2", "settled_3", "settled_9"] + [
        f"ewma_{s}" for s in spec["ewma_spans"]
    ]
    records = []
    for delay in (0, 5, 15):
        changed = labels.copy()
        changed["label_available_at"] = changed.calc_time + pd.Timedelta(minutes=delay)
        panel, _ = panel_for(ticker, changed, spec)
        records.append(
            dict(label_delay_minutes=delay, **compare_panels(baseline, panel, columns, spec))
        )
    save_json(
        output / "publication_delay_sensitivity.json",
        dict(
            identity_sha256=digest(identity),
            checks=records,
            scope="Exact features and validation/evaluation event-horizon membership",
        ),
    )
    exclusions = []
    for phase, start, end in (
        (
            "validation",
            spec["validation_start"],
            (pd.Timestamp(spec["test_start"]) - pd.Timedelta(nanoseconds=1)).isoformat(),
        ),
        ("evaluation", spec["test_start"], spec["test_end"]),
    ):
        folds = membership(baseline, start, end)
        admitted = set().union(*(test for _, _, test in folds))
        candidate = baseline[baseline.event_id.ge(start) & baseline.event_id.le(end)]
        missing = [key not in admitted for key in zip(candidate.event_id, candidate.horizon)]
        excluded = candidate.loc[missing, ["event_id", "horizon", "origin"]].copy()
        excluded["phase"], excluded["reason"] = phase, "origin_precedes_monthly_fit"
        exclusions.append(excluded)
    pd.concat(exclusions).to_csv(output / "fold_boundary_exclusions.csv", index=False)
    print(records, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run("sensitivities", parser.parse_args().config.resolve())


if __name__ == "__main__":
    main()

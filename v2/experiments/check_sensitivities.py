"""Offline publication-delay and temporal-boundary diagnostics for the fixed study."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from v2.experiments.run_study import (
    DEFAULT_CONFIG,
    ROOT,
    configuration,
    inputs,
    panel_for,
    save_json,
)
from v2.src.splits import monthly_folds


def main() -> None:
    spec = configuration(DEFAULT_CONFIG)
    output = ROOT / spec["output_dir"]
    ticker, labels, _, _ = inputs(spec)
    baseline, _ = panel_for(ticker, labels, spec)
    columns = ["settled_1", "settled_2", "settled_3", "settled_9"] + [
        f"ewma_{s}" for s in spec["ewma_spans"]
    ]
    reference = baseline.set_index(["event_id", "horizon"])
    records = []
    original_folds = [
        (cutoff, set(train.event_id), set(test.event_id))
        for cutoff, train, test in monthly_folds(baseline, spec["test_start"], spec["test_end"])
    ]
    for delay in (0, 5, 15):
        changed = labels.copy()
        changed["label_available_at"] = changed.calc_time + pd.Timedelta(minutes=delay)
        panel, _ = panel_for(ticker, changed, spec)
        candidate = panel.set_index(["event_id", "horizon"]).reindex(reference.index)
        same = np.isclose(reference[columns], candidate[columns], equal_nan=True).all()
        folds = [
            (cutoff, set(train.event_id), set(test.event_id))
            for cutoff, train, test in monthly_folds(panel, spec["test_start"], spec["test_end"])
        ]
        records.append(
            dict(
                label_delay_minutes=delay,
                same_origin_keys=panel.shape == baseline.shape
                and set(zip(panel.event_id, panel.horizon)) == set(reference.index),
                identical_settled_features=bool(same),
                identical_fold_membership=folds == original_folds,
            )
        )
    save_json(output / "publication_delay_sensitivity.json", records)
    admitted = set().union(*(test for _, _, test in original_folds))
    candidate = baseline[
        baseline.event_id.ge(spec["test_start"]) & baseline.event_id.le(spec["test_end"])
    ]
    excluded = candidate[~candidate.event_id.isin(admitted)][
        ["event_id", "horizon", "origin"]
    ].copy()
    excluded["reason"] = "origin_precedes_monthly_fit"
    excluded.to_csv(output / "fold_boundary_exclusions.csv", index=False)
    print(json.dumps(records, indent=2))
    print(f"Boundary exclusions: {excluded.event_id.nunique()} events / {len(excluded)} origins")


if __name__ == "__main__":
    main()

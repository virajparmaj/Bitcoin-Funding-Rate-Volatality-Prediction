"""Independent replay of the primary ridge folds and one RF fold from saved inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v2.experiments.run_study import DEFAULT_CONFIG, run, save_json
from v2.src.evaluation import predict_model
from v2.src.features import feature_columns
from v2.src.evaluation import horizon_folds
from v2.src.study_store import digest


def replay(output, spec, identity) -> None:
    times = [
        "event_id",
        "origin",
        "source_available_at",
        "label_available_at",
        "history_available_at",
        "calc_time",
    ]
    panel = pd.read_csv(
        output / "panel.csv.gz",
        parse_dates=times,
        date_format="mixed",
        float_precision="round_trip",
    )
    predictions = pd.read_csv(
        output / "predictions.csv.gz",
        parse_dates=times + ["fit_time"],
        date_format="mixed",
        float_precision="round_trip",
    )
    chosen = json.loads((output / "selection.json").read_text())["choices"]["4"]
    errors = []
    for index, (cutoff, train, test) in enumerate(horizon_folds(panel, spec, 4)):
        for model in ["ridge", "rf"] if index == 0 else ["ridge"]:
            replayed = predict_model(
                model, chosen[model], train, test, feature_columns("full"), spec
            )
            saved = predictions[
                predictions.variant.eq("main")
                & predictions.horizon.eq(4)
                & predictions.model.eq(model)
                & predictions.fit_time.eq(cutoff)
            ]
            actual = saved.set_index("event_id").reindex(test.event_id).prediction.to_numpy()
            np.testing.assert_allclose(replayed, actual, rtol=1e-9, atol=1e-12)
            errors.append(
                dict(
                    model=model,
                    fit_time=cutoff,
                    n=len(test),
                    max_abs_native_difference=float(np.max(abs(replayed - actual))),
                )
            )
    if not errors:
        raise ValueError("No forecast folds replayed")
    save_json(
        output / "forecast_replay.json",
        dict(
            status="passed",
            identity_sha256=digest(identity),
            checks=errors,
            scope="All four-hour ridge folds and first four-hour RF fold replayed from serialized panel; not a fresh independent dataset",
        ),
    )
    print(f"Replayed {len(errors)} model/fold combinations successfully")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run("replay", parser.parse_args().config.resolve())


if __name__ == "__main__":
    main()

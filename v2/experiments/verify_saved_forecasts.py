"""Independent replay of the primary ridge folds and one RF fold from saved inputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from v2.experiments.run_study import DEFAULT_CONFIG, ROOT, configuration, save_json
from v2.src.evaluation import predict_model
from v2.src.features import feature_columns
from v2.src.splits import monthly_folds


def main() -> None:
    spec = configuration(DEFAULT_CONFIG)
    output = ROOT / spec["output_dir"]
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
    panel = panel[panel.horizon.eq(4)]
    errors = []
    for index, (cutoff, train, test) in enumerate(
        monthly_folds(panel, spec["test_start"], spec["test_end"])
    ):
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
    save_json(
        output / "forecast_replay.json",
        dict(
            status="passed",
            checks=errors,
            scope="All four-hour ridge folds and first four-hour RF fold replayed from serialized panel; not a fresh independent dataset",
        ),
    )
    print(f"Replayed {len(errors)} model/fold combinations successfully")


if __name__ == "__main__":
    main()

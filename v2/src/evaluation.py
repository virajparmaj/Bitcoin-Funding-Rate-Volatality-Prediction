"""Validation-only selection followed by monthly frozen-parameter forecasting."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .baselines import baseline_predictions
from .features import feature_columns
from .forecast_models import estimator
from .splits import monthly_folds

HISTORY_COLUMNS = ["settled_1", "settled_2", "settled_3", "settled_9"]


def predict_model(model, params, train, test, columns, spec):
    """Fit preprocessing with training data only, in numerically stable bp units."""
    is_history = model == "history_ridge"
    target = train.target if is_history else train.residual
    fit = estimator(model, params, spec)
    fit.fit(train[columns], target * 1e4)
    prediction = fit.predict(test[columns]) / 1e4
    if not is_history:
        prediction = prediction + test.indication.to_numpy()
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite prediction from {model}")
    return prediction


def candidates(model: str, spec: dict) -> list[dict]:
    """Enumerate only the predeclared tuning grid."""
    if model in ("ridge", "history_ridge"):
        return [{"alpha": alpha} for alpha in spec["ridge_alphas"]]
    return [
        dict(max_depth=d, min_samples_leaf=n)
        for d, n in itertools.product(spec["rf_depths"], spec["rf_min_leaves"])
    ]


def tune(panel: pd.DataFrame, spec: dict):
    """Select on monthly 2022 forecasts only; return a complete candidate ledger."""
    choices, ledger = {}, []
    end = (pd.Timestamp(spec["test_start"]) - pd.Timedelta(nanoseconds=1)).isoformat()
    for horizon in spec["horizons"]:
        sample = panel[panel.horizon == horizon]
        validation = sample[sample.event_id.ge(spec["validation_start"]) & sample.event_id.le(end)]
        span = min(
            spec["ewma_spans"],
            key=lambda s: (validation.target - validation[f"ewma_{s}"]).abs().mean(),
        )
        choices[str(horizon)] = {"ewma_span": span}
        for model in ("ridge", "rf", "history_ridge"):
            columns = HISTORY_COLUMNS if model == "history_ridge" else feature_columns("full")
            scores = []
            for params in candidates(model, spec):
                errors = []
                for cutoff, train, test in monthly_folds(sample, spec["validation_start"], end):
                    if len(train) < spec["minimum_training_events"]:
                        raise ValueError("Insufficient training history")
                    prediction = predict_model(model, params, train, test, columns, spec)
                    errors.extend(abs(test.target.to_numpy() - prediction) * 1e4)
                score = float(np.mean(errors))
                if not errors or not np.isfinite(score):
                    raise ValueError("Empty/nonfinite validation loss")
                scores.append(score)
                ledger.append(
                    dict(horizon=horizon, model=model, params=params, mae_bp=score, n=len(errors))
                )
            choices[str(horizon)][model] = candidates(model, spec)[int(np.argmin(scores))]
            print(f"Selected h={horizon} {model}: {choices[str(horizon)][model]}", flush=True)
    return choices, ledger


def forecast(
    panel: pd.DataFrame,
    choices: dict,
    spec: dict,
    group="full",
    variant="main",
    include_models=True,
):
    """Generate identical-cohort baselines and models; failed fits abort visibly."""
    outputs, folds = [], []
    columns = feature_columns(group)
    for horizon in spec["horizons"]:
        sample = panel[panel.horizon == horizon]
        chosen = choices[str(horizon)]
        for cutoff, train, test in monthly_folds(sample, spec["test_start"], spec["test_end"]):
            if len(train) < spec["minimum_training_events"]:
                raise ValueError("Insufficient training history")
            predictions = baseline_predictions(train, test, chosen["ewma_span"])
            if include_models:
                for model in ("ridge", "rf", "history_ridge"):
                    cols = HISTORY_COLUMNS if model == "history_ridge" else columns
                    predictions[model] = predict_model(
                        model, chosen[model], train, test, cols, spec
                    )
            for model, prediction in predictions.items():
                if not np.isfinite(prediction).all():
                    raise ValueError(f"Nonfinite baseline: {model}")
                result = test[
                    [
                        "event_id",
                        "origin",
                        "source_available_at",
                        "source_row_id",
                        "age_minutes",
                        "label_available_at",
                        "calc_time",
                        "history_available_at",
                        "target",
                        "indication",
                        "terminal_proxy",
                        "terminal_proxy_at",
                        "horizon",
                    ]
                ].copy()
                result["prediction"], result["model"] = prediction, model
                result["variant"], result["fit_time"] = variant, cutoff
                result["feature_group"], result["units"] = group, "native_rate"
                outputs.append(result)
            folds.append(
                dict(
                    variant=variant,
                    horizon=horizon,
                    fit_time=cutoff,
                    train_n=len(train),
                    test_n=len(test),
                    max_train_label_available_at=train.label_available_at.max(),
                    min_test_origin=test.origin.min(),
                )
            )
        print(f"Forecasted {variant}, h={horizon}", flush=True)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(folds)

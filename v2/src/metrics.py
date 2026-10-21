"""Paired losses and calendar-block uncertainty; no fee-to-MAE profit inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def loss_summary(frame, target="target") -> dict:
    """Score fixed predictions against an explicit target and indication baseline."""
    error = (frame.prediction - frame[target]).to_numpy() * 1e4
    baseline = (frame.indication - frame[target]).to_numpy() * 1e4
    denominator = np.sum(baseline**2)
    return dict(
        n=len(frame),
        mae_bp=float(np.mean(abs(error))),
        rmse_bp=float(np.sqrt(np.mean(error**2))),
        bias_bp=float(np.mean(error)),
        gain_bp=float(np.mean(abs(baseline) - abs(error))),
        mse_skill=float(1 - np.sum(error**2) / denominator) if denominator > 0 else None,
    )


def bootstrap_gain(events, gains, draws=5000, block_days=7, seed=42):
    """Paired moving calendar blocks, retaining empty days and event weights.

    Returns conditional uncertainty for saved forecasts, not training uncertainty.
    Resample complete blocks plus a truncated final block to preserve calendar size.
    """
    values = pd.DataFrame({"day": pd.to_datetime(events, utc=True).dt.floor("D"), "gain": gains})
    daily = values.groupby("day").gain.agg(["sum", "count"])
    daily = daily.reindex(
        pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0
    )
    days = len(daily)
    if days < 2 * block_days or len(values) < 2:
        return dict(gain_low_bp=None, gain_high_bp=None, p_value=None, sufficient_blocks=False)
    array = daily[["sum", "count"]].to_numpy()
    cumulative = np.vstack([np.zeros(2), array.cumsum(axis=0)])
    full, remainder = divmod(days, block_days)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, days - block_days + 1, size=(draws, full))
    totals = (cumulative[starts + block_days] - cumulative[starts]).sum(axis=1)
    if remainder:
        tail = rng.integers(0, days - remainder + 1, size=draws)
        totals += cumulative[tail + remainder] - cumulative[tail]
    valid = totals[:, 1] > 0
    samples = totals[valid, 0] / totals[valid, 1]
    estimate = float(np.mean(gains))
    p_value = (1 + np.sum(abs(samples - estimate) >= abs(estimate))) / (len(samples) + 1)
    return dict(
        gain_low_bp=float(np.quantile(samples, 0.025)),
        gain_high_bp=float(np.quantile(samples, 0.975)),
        p_value=float(p_value),
        sufficient_blocks=True,
    )


def inference_table(predictions, spec):
    """Joint calendar sample for horizons/models; Holm-adjust secondary comparisons."""
    rows = []
    for (variant, horizon, model), frame in predictions.groupby(["variant", "horizon", "model"]):
        gains = (
            abs(frame.target - frame.indication) - abs(frame.target - frame.prediction)
        ).to_numpy() * 1e4
        for block in spec["block_days"]:
            row = dict(
                variant=variant,
                horizon=horizon,
                model=model,
                block_days=block,
                **loss_summary(frame),
            )
            row.update(
                bootstrap_gain(frame.event_id, gains, spec["bootstrap_draws"], block, spec["seed"])
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    secondary = (
        result.block_days.eq(7)
        & result.model.isin(["ridge", "rf"])
        & ~(
            result.variant.eq("main")
            & result.horizon.eq(spec["primary_horizon"])
            & result.model.eq("ridge")
        )
    )
    valid = secondary & result.p_value.notna()
    result.loc[valid, "p_holm_secondary"] = multipletests(
        result.loc[valid, "p_value"], method="holm"
    )[1]
    return result

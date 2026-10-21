"""Small causal feature sets with explicit latest-source availability lineage."""

from __future__ import annotations

import numpy as np
import pandas as pd

INDICATION = ["indication", "age_minutes", "event_hour"]
HISTORY = ["event_change", "event_std", "settled_1", "settled_2", "settled_3", "settled_9"]
MARKET = ["mark_index_spread", "price_return", "oi_change"]


def feature_columns(group: str) -> list[str]:
    """Return the declared feature set; unknown groups must fail."""
    if group not in ("indication", "history", "full"):
        raise ValueError(f"Unknown feature group: {group}")
    return (
        INDICATION
        + (HISTORY if group != "indication" else [])
        + (MARKET if group == "full" else [])
    )


def ticker_features(ticker: pd.DataFrame) -> pd.DataFrame:
    """Derive histories using only current/earlier reduced rows, without fills."""
    data = ticker.copy()
    data["event_change"] = data.groupby("event_id").funding_rate.diff()
    data["event_std"] = data.groupby("event_id").funding_rate.transform(
        lambda s: s.expanding(min_periods=2).std()
    )
    data["mark_index_spread"] = data.mark_price / data.index_price.replace(0, np.nan) - 1
    elapsed = data.arrival.diff().dt.total_seconds()
    data["price_return"] = data.mark_price / data.mark_price.shift(1).replace(0, np.nan) - 1
    data["oi_change"] = data.open_interest.diff()
    data.loc[~elapsed.between(0, 90 * 60), ["price_return", "oi_change"]] = np.nan
    data["event_hour"] = data.event_id.dt.hour
    return data.replace([np.inf, -np.inf], np.nan)


def add_settled_history(panel: pd.DataFrame, labels: pd.DataFrame, spans: list[int]):
    """Join only published labels; never use the forthcoming event's settlement."""
    history = labels.sort_values("label_available_at").copy()
    for lag in (1, 2, 3, 9):
        history[f"settled_{lag}"] = history.target.shift(lag - 1)
    for span in spans:
        history[f"ewma_{span}"] = history.target.ewm(span=span, adjust=False).mean()
    columns = [f"settled_{lag}" for lag in (1, 2, 3, 9)] + [f"ewma_{s}" for s in spans]
    history = history[["label_available_at"] + columns].rename(
        columns={"label_available_at": "history_available_at"}
    )
    result = pd.merge_asof(
        panel.sort_values("origin"),
        history,
        left_on="origin",
        right_on="history_available_at",
        direction="backward",
    )
    result["residual"] = result.target - result.indication
    return result.sort_values(["event_id", "horizon"]).reset_index(drop=True)

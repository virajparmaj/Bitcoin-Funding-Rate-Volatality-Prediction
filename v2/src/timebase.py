"""Preserve real receipt times and row identity; never manufacture an hourly clock."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_ticker(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize usable rows and account for exclusions without imputing rates."""
    data = raw.copy()
    data["source_row_id"] = np.arange(len(data))
    data["arrival"] = pd.to_datetime(data.local_timestamp, unit="us", utc=True)
    data["event_id"] = pd.to_datetime(data.funding_timestamp, unit="us", utc=True)
    valid = data.arrival.notna() & data.event_id.notna() & np.isfinite(data.funding_rate)
    rejected = data.loc[~valid, ["source_row_id"]].assign(reason="missing_time_event_or_rate")
    data = data.loc[valid].sort_values(["arrival", "source_row_id"]).copy()
    # Conflicting same-time event updates cannot be ordered from this reduction.
    duplicate = data.duplicated(["event_id", "arrival"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate event/arrival keys require source reconciliation")
    return data, rejected

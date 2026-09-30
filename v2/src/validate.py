"""Scientific invariants: availability and identity rather than feature-name bans."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_panel(panel: pd.DataFrame, max_age_minutes: float) -> None:
    """Reject unavailable observations, labels and inconsistent origin keys."""
    if panel.empty or panel.duplicated(["event_id", "horizon"]).any():
        raise ValueError("Empty panel or duplicate event/horizon")
    if not np.isfinite(panel[["target", "indication"]].to_numpy()).all():
        raise ValueError("Nonfinite label/indication")
    if not (panel.source_available_at <= panel.origin).all():
        raise ValueError("Future source observation")
    if not panel.age_minutes.between(0, max_age_minutes).all():
        raise ValueError("Invalid observation age")
    expected = panel.event_id - pd.to_timedelta(panel.horizon, unit="h")
    if not expected.equals(panel.origin):
        raise ValueError("Origin does not match event/horizon")
    if not (panel.label_available_at >= panel.event_id).all():
        raise ValueError("Settlement label precedes scheduled event")
    if "history_available_at" in panel:
        known = panel.history_available_at.notna()
        if not (panel.loc[known, "history_available_at"] <= panel.loc[known, "origin"]).all():
            raise ValueError("Future settled-history feature")

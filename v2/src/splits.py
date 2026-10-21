"""Monthly expanding folds that respect both label maturity and decision origins."""

from __future__ import annotations

import pandas as pd


def monthly_folds(panel: pd.DataFrame, start: str, end: str):
    """Keep an event's horizons together and purge pre-fit boundary origins.

    An event at month start can have origins in the previous month. Such boundary
    events are excluded from that month's evaluation, never scored with a model
    fitted after their forecast origins.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for cutoff in pd.date_range(start, end, freq="MS"):
        stop = cutoff + pd.offsets.MonthBegin(1)
        train = panel.label_available_at.lt(cutoff) & panel.event_id.lt(cutoff)
        earliest = panel.groupby("event_id").origin.transform("min")
        test = panel.event_id.ge(cutoff) & panel.event_id.lt(stop) & panel.event_id.le(end)
        test &= earliest.ge(cutoff)
        if test.any():
            yield cutoff, panel.loc[train], panel.loc[test]

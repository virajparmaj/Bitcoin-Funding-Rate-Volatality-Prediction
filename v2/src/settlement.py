"""Construct event-specific backward-only origins with auditable availability."""

from __future__ import annotations

import pandas as pd


def build_origin_panel(ticker, labels, horizons, max_age_minutes, delay_minutes=0):
    """Return eligible rows and one exclusion reason per rejected event/horizon."""
    rows, excluded = [], []
    lookup = labels.set_index("event_id")
    for event, group in ticker.groupby("event_id", sort=True):
        available = group.arrival + pd.Timedelta(minutes=delay_minutes)
        terminal = group.loc[group.arrival < event]
        proxy = terminal.funding_rate.iloc[-1] if len(terminal) else float("nan")
        for horizon in horizons:
            origin = event - pd.Timedelta(hours=horizon)
            eligible = group.loc[available <= origin]
            reason = None
            if event not in lookup.index:
                reason = "missing_settlement_label"
            elif eligible.empty:
                reason = "no_same_event_observation_at_origin"
            else:
                last = eligible.iloc[-1]
                at = last.arrival + pd.Timedelta(minutes=delay_minutes)
                age = (origin - at).total_seconds() / 60
                if age > max_age_minutes:
                    reason = "stale_observation"
            if reason:
                excluded.append({"event_id": event, "horizon": horizon, "reason": reason})
                continue
            label = lookup.loc[event]
            row = last.to_dict()
            row.update(
                origin=origin,
                horizon=horizon,
                source_available_at=at,
                age_minutes=age,
                target=label.target,
                indication=last.funding_rate,
                calc_time=label.calc_time,
                label_available_at=label.label_available_at,
                terminal_proxy=proxy,
                terminal_proxy_at=terminal.arrival.iloc[-1],
                effective_lead_hours=(event - last.arrival).total_seconds() / 3600,
            )
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(excluded, columns=["event_id", "horizon", "reason"])

"""Measurement helpers for the v1 audit.

Separate from :mod:`legacy`, which reproduces v1 *behaviour*. Nothing here reimplements
v1; these functions measure properties of the shipped data and the shipped result CSVs.
They are read-only and have no side effects.

Used by ``notebooks/00_audit_v1.ipynb``. Later phases reuse
:func:`settlement_series` and :func:`stationarity_summary` for comparison against the
corrected settlement panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller

from . import config

# ===========================================
# Scoring
# ===========================================


def score_predictors(actual: pd.Series, candidates: dict[str, pd.Series]) -> pd.DataFrame:
    """Score candidate predictors against a common target.

    Args:
        actual: Realised target series.
        candidates: Mapping of predictor name to predicted series.

    Returns:
        Frame indexed by predictor name with ``r2``, ``mse``, and ``mae_bp``,
        sorted best-first by R-squared.
    """
    rows = {
        name: {
            "r2": r2_score(actual, values),
            "mse": mean_squared_error(actual, values),
            "mae_bp": float((actual - values).abs().mean() * config.BP),
        }
        for name, values in candidates.items()
    }
    return pd.DataFrame(rows).T.sort_values("r2", ascending=False)


def constant_column_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Report cardinality of columns that are supposed to vary.

    Args:
        df: Frame to inspect, typically a shipped predictions CSV.
        columns: Column names to check.

    Returns:
        Frame indexed by column with ``nunique`` and the observed ``values``.
    """
    rows = {
        name: {
            "nunique": int(df[name].nunique()),
            "values": np.sort(df[name].unique())[:3].tolist(),
        }
        for name in columns
        if name in df.columns
    }
    return pd.DataFrame(rows).T


# ===========================================
# Timebase
# ===========================================


def timestamp_drift(raw: pd.DataFrame, synthetic: pd.DatetimeIndex) -> pd.DataFrame:
    """Measure drift of v1's synthetic clock against the real ``local_timestamp``.

    Args:
        raw: Raw frame as shipped.
        synthetic: Synthetic clock from :func:`legacy.synthetic_clock`.

    Returns:
        Single-row frame with drift at the first and last row (hours and days), the
        probability that the synthetic hour-of-day equals the real hour-of-day, and the
        number of distinct gaps in the synthetic grid.
    """
    real = pd.to_datetime(raw["local_timestamp"], unit="us", utc=True)
    drift_hours = (synthetic - real).dt.total_seconds() / 3600.0
    return pd.DataFrame(
        [
            {
                "drift_first_row_h": drift_hours.iloc[0],
                "drift_last_row_h": drift_hours.iloc[-1],
                "drift_last_row_days": drift_hours.iloc[-1] / 24.0,
                "p_hour_of_day_matches": float((synthetic.hour == real.dt.hour).mean()),
                "unique_synthetic_gaps_h": int(synthetic.to_series().diff().nunique()),
            }
        ]
    )


def backfill_audit(raw: pd.DataFrame) -> pd.DataFrame:
    """Count cells that ``data_processing.py:75`` fills from *future* observations.

    A cell is backward-filled when it is still NaN after ``ffill`` but not after
    ``bfill``. Columns that are 100% null cannot be filled in either direction and are
    excluded, since ``bfill`` does not in fact touch them.

    Args:
        raw: Raw frame as shipped.

    Returns:
        Frame indexed by column with ``backward_filled`` counts, sorted descending.
    """
    ordered = raw.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], unit="us", utc=True)
    ordered = ordered.sort_values("timestamp").reset_index(drop=True)

    still_na_after_ffill = ordered.ffill().isna().sum()
    counts = still_na_after_ffill[~ordered.isna().all()]
    counts = counts[counts > 0].sort_values(ascending=False)
    return counts.rename("backward_filled").to_frame()


def settlement_series(raw: pd.DataFrame, how: str = "last") -> pd.Series:
    """Collapse the hourly file to one observation per funding settlement.

    Args:
        raw: Raw frame as shipped.
        how: Aggregation within a settlement — ``last``, ``first``, or ``mean``.

    Returns:
        Funding rate indexed by settlement timestamp (UTC), ascending.
    """
    grouped = raw.dropna(subset=["funding_timestamp"]).groupby("funding_timestamp")["funding_rate"]
    series = getattr(grouped, how)().dropna()
    series.index = pd.to_datetime(series.index, unit="us", utc=True)
    return series.sort_index()


# ===========================================
# Structure and stationarity
# ===========================================


def dataset_structure(raw: pd.DataFrame) -> pd.DataFrame:
    """Characterise the shipped file's sampling, ties, and persistence.

    The tie share matters because ``data_processing.py:105`` builds the direction label
    with a strict ``>``, so every settlement where the rate is unchanged between
    consecutive hourly rows is silently coded as class 0 ("down").

    Args:
        raw: Raw frame as shipped.

    Returns:
        Single-row frame of structural statistics.
    """
    funding = raw["funding_rate"]
    forward = funding.shift(-1)
    direction = (forward > funding).astype(int).iloc[:-1]
    rho = funding.autocorr(1)
    settlements = pd.to_datetime(
        raw["funding_timestamp"].dropna().unique(), unit="us", utc=True
    ).sort_values()
    gaps = pd.Series(settlements).diff().dt.total_seconds() / 3600.0

    return pd.DataFrame(
        [
            {
                "n_rows": len(raw),
                "n_settlements": int(raw["funding_timestamp"].nunique()),
                "modal_settlement_gap_h": float(gaps.mode().iloc[0]),
                "tie_share_pct": float((forward == funding).mean() * 100),
                "direction_up_pct": float(direction.mean() * 100),
                "direction_down_pct": float((1 - direction.mean()) * 100),
                "autocorr_lag1": rho,
                "autocorr_lag8": funding.autocorr(config.SETTLEMENT_HOURS),
                "autocorr_lag24": funding.autocorr(24),
                "ar1_effective_n": len(funding) * (1 - rho) / (1 + rho),
            }
        ]
    )


def stationarity_summary(series_by_name: dict[str, pd.Series]) -> pd.DataFrame:
    """Run the Augmented Dickey-Fuller test on several series at once.

    Args:
        series_by_name: Mapping of label to series.

    Returns:
        Frame indexed by label with ``n``, ``adf_stat``, and ``p_value``.
    """
    rows = {}
    for name, series in series_by_name.items():
        clean = series.dropna()
        statistic, p_value = adfuller(clean)[:2]
        rows[name] = {"n": len(clean), "adf_stat": statistic, "p_value": p_value}
    return pd.DataFrame(rows).T


def garch_aic_units_shift(n_observations: int, scaling_factor: float) -> pd.DataFrame:
    """Show how much of a reported AIC is explained purely by rescaling the data.

    Rescaling a series by ``c`` multiplies the density by ``1/c`` at every point, so the
    log-likelihood shifts by ``-n * ln(c)`` and AIC by ``+2n * ln(c)`` for an otherwise
    identical fit. An AIC that moves when the units move is not evidence of fit quality.

    Args:
        n_observations: Sample size used to fit the model.
        scaling_factor: Multiplicative rescaling applied to the series.

    Returns:
        Single-row frame with the reported AIC, the shift, and the rescaled equivalent.
    """
    shift = 2 * n_observations * np.log(scaling_factor)
    return pd.DataFrame(
        [
            {
                "reported_aic": config.V1_GARCH_AIC,
                "n": n_observations,
                "scaling_factor": scaling_factor,
                "aic_shift_2n_ln_c": shift,
                "aic_after_rescaling": config.V1_GARCH_AIC + shift,
            }
        ]
    )

"""Faithful reimplementation of v1 logic — FOR AUDIT ONLY.

This module exists because ``utilities/`` cannot be imported. ``utilities/__init__.py``
star-imports ``data_pull`` at package-init time, which imports ``tardis_dev``; and
``utilities/functions.py`` imports ``imblearn`` at module scope. Neither is present in
v1's ``requirements.txt``, so ``from utilities.data_processing import process_pipeline``
raises ``ModuleNotFoundError`` before any v1 function can be called. Patching
``utilities/`` is out of scope: the v1 tree is the evidence under audit and must stay
byte-for-byte intact.

Every function below therefore reproduces v1 behaviour *including its defects*, with the
originating ``file:line`` cited in the docstring. Nothing here is used by v2 modelling
code. v2 features live in ``features.py``; v2 baselines live in ``baselines.py``.

Style note: v1 mutates frames in place. These reimplementations copy first and return new
objects, which is observationally identical for the audited call sequences and keeps the
module compliant with the project's immutability rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  # v1 only — banned in v2 code

from . import config

# ===========================================
# Data loading and preprocessing
# ===========================================


def load_v1_raw() -> pd.DataFrame:
    """Load the shipped normalized dataset exactly as v1 does.

    Mirrors ``utilities/data_processing.py:24`` (``load_data``), minus the bare
    ``except Exception`` that swallowed load failures into a ``None`` return.

    Returns:
        The raw frame, 42,179 rows x 11 columns, untouched.
    """
    return pd.read_csv(config.V1_DATA_CSV)


def preprocess_v1(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce ``utilities/data_processing.py:47`` (``preprocess_data``).

    Defect preserved: ``df.bfill()`` at ``data_processing.py:75`` runs on the whole
    frame before any train/test split, so gaps are filled from future observations.

    Args:
        df: Raw frame from :func:`load_v1_raw`.

    Returns:
        A new frame with the v1 timestamp conversion, ffill, and bfill applied.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="us", utc=True)
    out = out.sort_values("timestamp")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill()
    out = out.bfill()  # <- data_processing.py:75, look-ahead
    return out.reset_index(drop=True)


def remove_outliers_v1(
    series: pd.Series, z_score_threshold: float = config.V1_Z_SCORE_THRESHOLD
) -> pd.Series:
    """Reproduce ``utilities/functions.py:252`` (``remove_outliers``).

    Two defects preserved:
      1. z-scores are computed on the full sample, so the outlier mask is informed by
         data the model has not yet seen at prediction time;
      2. ``np.where`` yields *positions* within the ``dropna()``'d series, which are then
         applied via ``.iloc`` to the original series — an index misalignment whenever
         the series contains any NaN.

    Args:
        series: Funding-rate series.
        z_score_threshold: Absolute z-score above which a point is dropped.

    Returns:
        A shortened series; the dropped positions become NaN once reassigned to a frame.
    """
    z_scores = np.abs(stats.zscore(series.dropna()))
    filtered_indices = np.where(z_scores < z_score_threshold)
    return series.iloc[filtered_indices].copy()


def add_lag_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce ``utilities/functions.py:33`` (``add_lag_features``).

    Defect preserved: the whole-frame ``df.fillna(0, inplace=True)`` at
    ``functions.py:64`` converts every NaN — including outliers nulled by
    :func:`remove_outliers_v1` — to exactly ``0.0``.

    Args:
        df: Preprocessed frame.

    Returns:
        A new frame with lag columns added and all NaNs replaced by zero.
    """
    out = df.copy()
    out["funding_rate_lag1"] = out["funding_rate"].shift(1)
    out["funding_rate_lag2"] = out["funding_rate"].shift(2)
    out["open_interest_lag1"] = out["open_interest"].shift(1)
    out["mark_price_lag1"] = out["mark_price"].shift(1)
    return out.fillna(0)  # <- functions.py:64


def add_technical_indicators_v1(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce ``utilities/functions.py:66`` (``add_technical_indicators``).

    Defect preserved: ``pct_change`` at ``functions.py:93-97`` is applied to a series
    that oscillates through zero, so the ratio explodes and flips sign near crossings.

    Args:
        df: Frame carrying ``funding_rate``, ``mark_price``, ``open_interest``.

    Returns:
        A new frame with v1's moving averages, EWMAs, rolling volatility, and
        rate-of-change columns.
    """
    out = df.copy()
    ma, vol = config.V1_MA_WINDOW, config.V1_VOL_WINDOW
    out[f"funding_rate_ma{ma}"] = out["funding_rate"].rolling(window=ma).mean()
    out["funding_rate_ma5"] = out["funding_rate"].rolling(window=5).mean()
    out[f"funding_rate_ema{ma}"] = out["funding_rate"].ewm(span=ma, adjust=False).mean()
    out["funding_rate_ema5"] = out["funding_rate"].ewm(span=5, adjust=False).mean()
    out["volatility_5h"] = out["mark_price"].rolling(window=vol).std()
    out["funding_rate_roc1"] = out["funding_rate"].pct_change(periods=1)  # <- :93
    out["funding_rate_roc3"] = out["funding_rate"].pct_change(periods=3)  # <- :94
    out["open_interest_roc"] = out["open_interest"].pct_change(periods=1)  # <- :97
    return out


# ===========================================
# Analysis A (notebooks/stat429_analysis_a.ipynb)
# ===========================================

#: Predictors used by ``stat429_analysis_a.ipynb`` code cell 8 (JSON cell index 22).
ANALYSIS_A_BASE_FEATURES: tuple[str, ...] = (
    "open_interest",
    "mark_price",
    "hour",
    "day",
    "month",
)


def analysis_a_frame(preprocessed: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the Analysis A feature block, JSON cell 22 (code cell 8).

    Builds the calendar features from ``local_timestamp`` and the 24-hour rolling
    standard deviation. Defect preserved: ``std`` is a *centred-on-now* window — it is
    computed over ``t-23 .. t``, so it contains the target ``funding_rate(t)`` itself.

    Args:
        preprocessed: Output of :func:`preprocess_v1`.

    Returns:
        A new frame with ``hour``, ``day``, ``month``, ``std``, plus the lagged and
        forward-shifted columns needed by the corrected variants.
    """
    out = preprocessed.copy()
    out["local_timestamp"] = pd.to_datetime(out["local_timestamp"], unit="us")
    out["hour"] = out["local_timestamp"].dt.hour
    out["day"] = out["local_timestamp"].dt.day
    out["month"] = out["local_timestamp"].dt.month

    window = config.V1_ROLLING_STD_WINDOW
    rolling_std = out["funding_rate"].rolling(window=window).std()
    out["std"] = rolling_std  # contains funding_rate(t)
    out["std_lagged"] = rolling_std.shift(1)  # ends at t-1, point-in-time safe
    out["funding_rate_fwd1"] = out["funding_rate"].shift(-1)
    return out


def analysis_a_variant(
    frame: pd.DataFrame,
    *,
    chronological: bool = False,
    lag_rolling_std: bool = False,
    forecast_target: bool = False,
) -> dict[str, float]:
    """Fit Analysis A's Random Forest under one of the four audit variants.

    Variant 1 (all flags ``False``) is v1 exactly as written: contemporaneous target,
    look-ahead rolling std, and ``train_test_split(random_state=42)`` on a series with
    lag-1 autocorrelation 0.9887.

    Args:
        frame: Output of :func:`analysis_a_frame`.
        chronological: Split by time order instead of shuffling.
        lag_rolling_std: Use the ``t-1``-terminated rolling std.
        forecast_target: Predict ``funding_rate(t+1)`` instead of ``funding_rate(t)``.

    Returns:
        Mapping with ``r2``, ``mse``, ``n_train``, ``n_test``.
    """
    std_col = "std_lagged" if lag_rolling_std else "std"
    target = "funding_rate_fwd1" if forecast_target else "funding_rate"
    columns = list(ANALYSIS_A_BASE_FEATURES) + [std_col]

    subset = frame[columns + [target]].dropna()
    features, response = subset[columns], subset[target]

    if chronological:
        split = int((1 - config.V1_TEST_SIZE) * len(features))
        x_train, x_test = features.iloc[:split], features.iloc[split:]
        y_train, y_test = response.iloc[:split], response.iloc[split:]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            features, response, test_size=config.V1_TEST_SIZE, random_state=config.SEED
        )

    model = RandomForestRegressor(n_estimators=config.V1_N_ESTIMATORS, random_state=config.SEED)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    return {
        "r2": r2_score(y_test, predicted),
        "mse": mean_squared_error(y_test, predicted),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }


# ===========================================
# Model 3 target reconstruction
# ===========================================


def closed_form_funding_rate(predictions: pd.DataFrame) -> pd.Series:
    """Recover ``funding_rate(t)`` from three shipped Model 3 features.

    ``model_development.ipynb`` cell 14 selects features by exclusion::

        features = [c for c in df.columns
                    if c not in ['funding_rate', 'future_funding_rate', 'direction']]

    That keeps ``funding_rate_ma3``, ``funding_rate_lag1``, and ``funding_rate_lag2`` in
    the design matrix. Since ``ma3(t) = (fr(t) + fr(t-1) + fr(t-2)) / 3``::

        fr(t) = 3 * ma3(t) - lag1(t) - lag2(t)

    so the excluded target's own contemporaneous value is exactly recoverable by a linear
    combination the Random Forest is free to approximate.

    Args:
        predictions: A shipped v1 predictions CSV.

    Returns:
        The reconstructed contemporaneous funding rate.
    """
    return (
        3 * predictions["funding_rate_ma3"]
        - predictions["funding_rate_lag1"]
        - predictions["funding_rate_lag2"]
    )


# ===========================================
# Timebase (dgieser3/research.ipynb)
# ===========================================


def synthetic_clock(n_rows: int) -> pd.DatetimeIndex:
    """Reproduce the synthetic timestamp of ``dgieser3/research.ipynb`` cell 2.

    v1 overwrote the real clock with ``datetime(2020,1,1) + index * timedelta(hours=1)``
    and saved that to ``binance_btc_perp.csv`` as the ``timestamp`` column, asserting a
    perfectly regular hourly grid the exchange data does not have.

    Args:
        n_rows: Number of rows to stamp.

    Returns:
        The synthetic hourly index, UTC.
    """
    return pd.DatetimeIndex(
        config.V1_SYNTHETIC_EPOCH + pd.to_timedelta(np.arange(n_rows), unit="h")
    )


def zeroed_outlier_mask(preprocessed: pd.DataFrame) -> pd.Series:
    """Locate the rows ``process_pipeline(handle_outliers=True)`` rewrites as ``0.0``.

    v1 assigns the *shortened* output of :func:`remove_outliers_v1` back onto the full
    frame. The dropped rows become NaN, and ``functions.py:64`` then rewrites them as
    ``0.0`` — a funding rate of zero is a valid, meaningful value, not a missing-data
    marker. Those zeros subsequently feed every lag, moving average, GARCH input, and
    direction label.

    Args:
        preprocessed: Output of :func:`preprocess_v1`.

    Returns:
        Boolean mask, ``True`` where the true funding rate was replaced by zero.
    """
    kept = remove_outliers_v1(preprocessed["funding_rate"])
    return preprocessed.assign(funding_rate=kept)["funding_rate"].isna()

"""
Feature engineering and utility functions for the Bitcoin Funding Rate Volatility Prediction project.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import MODEL1_RF_PATH, MODEL2_GARCH_PATH, SCALER1_RF_PATH
from utilities.model_utils import load_model, load_garch_model

# ===========================================
# Feature Engineering Functions
# ===========================================


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features to capture temporal trends.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with lag features added
        
    Raises:
        KeyError: If 'funding_rate' column is missing
    """
    if 'funding_rate' not in df.columns:
        raise KeyError("'funding_rate' column is missing. Check the input data.")

    # Lagged funding rate features
    df['funding_rate_lag1'] = df['funding_rate'].shift(1)
    df['funding_rate_lag2'] = df['funding_rate'].shift(2)

    if 'open_interest' in df.columns:
        df['open_interest_lag1'] = df['open_interest'].shift(1)
    else:
        df['open_interest_lag1'] = np.nan

    if 'mark_price' in df.columns:
        df['mark_price_lag1'] = df['mark_price'].shift(1)
    else:
        df['mark_price_lag1'] = np.nan

    # Handle missing values from lagging
    df.fillna(0, inplace=True)
    return df

def add_technical_indicators(df: pd.DataFrame, ma_window: int = 3, vol_window: int = 5) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.
    
    Args:
        df: Input DataFrame
        ma_window: Window size for moving average
        vol_window: Window size for volatility calculation
        
    Returns:
        DataFrame with technical indicators added
        
    Raises:
        KeyError: If required columns are missing
    """
    if 'funding_rate' not in df.columns or 'mark_price' not in df.columns:
        raise KeyError("'funding_rate' or 'mark_price' missing.")

    df[f'funding_rate_ma{ma_window}'] = df['funding_rate'].rolling(window=ma_window).mean()
    df['funding_rate_ma5'] = df['funding_rate'].rolling(window=5).mean()
    
    df[f'funding_rate_ema{ma_window}'] = df['funding_rate'].ewm(span=ma_window, adjust=False).mean()
    df['funding_rate_ema5'] = df['funding_rate'].ewm(span=5, adjust=False).mean()

    df['volatility_5h'] = df['mark_price'].rolling(window=vol_window).std()

    df['funding_rate_roc1'] = df['funding_rate'].pct_change(periods=1)
    df['funding_rate_roc3'] = df['funding_rate'].pct_change(periods=3)
    
    if 'open_interest' in df.columns:
        df['open_interest_roc'] = df['open_interest'].pct_change(periods=1)
    else:
        df['open_interest_roc'] = np.nan
        
    return df

def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms to capture relationships between features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with interaction terms added
        
    Raises:
        KeyError: If required columns are missing
    """
    if 'funding_rate_lag1' not in df.columns or 'funding_rate_lag2' not in df.columns:
        raise KeyError("'funding_rate_lag1' or 'funding_rate_lag2' columns are missing. Ensure lag features are added first.")
    if 'funding_rate_ma3' not in df.columns:
        raise KeyError("'funding_rate_ma3' column is missing. Ensure technical indicators are added first.")

    # Interaction Terms
    df['interaction1'] = df['funding_rate_lag1'] * df['funding_rate_lag2']

    # Handle potential division-by-zero issues for interaction2
    interaction2 = df['funding_rate_ma3'] / (df['funding_rate_lag1'].replace(0, np.nan) + 1e-6)
    interaction2 = interaction2.replace([np.inf, -np.inf], np.nan)
    interaction2 = interaction2.fillna(0)
    df['interaction2'] = interaction2

    if 'mark_price_lag1' in df.columns and 'funding_rate_ma3' in df.columns:
        df['interaction3'] = df['mark_price_lag1'] * df['funding_rate_ma3']

    return df

# ===========================================
# Model 1 & Model 2 Integration for Model 3
# ===========================================

def add_model1_direction(df):
    # Load model and scaler
    model1 = load_model(MODEL1_RF_PATH)
    scaler1 = load_model(SCALER1_RF_PATH) 

    model1_feature_columns = [
        'funding_rate_lag1', 'funding_rate_lag2', 'funding_rate_ma3', 'funding_rate_ma5', 'funding_rate_ema3',
        'open_interest', 'open_interest_lag1', 'open_interest_roc',
        'mark_price', 'mark_price_lag1', 'volatility_5min',
        'funding_rate_roc1', 'funding_rate_roc3', 'interaction2', 'interaction3'
    ]

    missing_columns = [col for col in model1_feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for Model 1 prediction: {missing_columns}")

    X = df[model1_feature_columns]

    # Final cleanup step to remove infinities and NaNs before scaling
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.clip(-1e9, 1e9)
    print("Max values after clipping:\n", X.max())
    print("Min values after clipping:\n", X.min())

    X_scaled = scaler1.transform(X)

    # Predict direction
    df['model1_direction_pred'] = model1.predict(X_scaled)
    return df

def add_model2_volatility(df, model2_result):
    """
    Instead of forecasting 5 steps at the end, 
    we retrieve the entire in-sample or one-step-ahead 
    forecast for each time index used during GARCH training.
    """
    # Suppose your GARCH model was fitted on an aligned index
    # 'in_sample' or 'one_step_forecast' approach:
    garch_forecast = model2_result.forecast(reindex=False)  # reindex to match the original
    # garch_forecast.variance is a DataFrame with the same index used in training

    # Ensure the indices align with df. If not, you may need to reindex or merge
    volatility_series = garch_forecast.variance.iloc[:, 0]
    # Now each timestamp has a forecasted variance

    # Merge or align with df:
    # If your df has the same datetime index as garch_forecast,
    # you can do something like:
    df['model2_volatility_h1'] = volatility_series.reindex(df.index, fill_value=np.nan)

    # Then fill or drop NaN as needed
    return df

# ===========================================
# Data Sampling and Balancing
# ===========================================

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: Union[float, Dict] = 1.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance the classes in the training data.

    Args:
        X_train: Training features
        y_train: Training target variable
        sampling_strategy: Sampling strategy for SMOTE
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_resampled, y_resampled): Balanced training data
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def apply_smote_tomek(X_train, y_train, random_state=42):
    """
    Apply SMOTE-Tomek to balance the classes in the training data.
    """
    from imblearn.combine import SMOTETomek
    smote_tomek = SMOTETomek(random_state=random_state)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

# ===========================================
# Time Series Diagnostics
# ===========================================

def perform_ljung_box_test(residuals, lags=10):
    """
    Perform the Ljung-Box test on residuals.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    print(lb_test)

def plot_acf_pacf(series, lags=50):
    """
    Plot ACF and PACF plots.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_acf(series.dropna(), lags=lags, ax=plt.gca())
    plt.subplot(1, 2, 2)
    plot_pacf(series.dropna(), lags=lags, ax=plt.gca())
    plt.show()

def remove_outliers(series, z_score_threshold=3):
    """
    Remove outliers from a pandas Series based on z-score threshold.
    """
    z_scores = np.abs(stats.zscore(series.dropna()))
    filtered_indices = np.where(z_scores < z_score_threshold)
    return series.iloc[filtered_indices].copy()

def winsorize_series(series, limits):
    """
    Winsorize a pandas Series to handle extreme values.
    """
    from scipy.stats.mstats import winsorize
    return winsorize(series, limits=limits)

def rescale_series(series, scaling_factor):
    """
    Rescale a pandas Series by a scaling factor.
    """
    return series * scaling_factor
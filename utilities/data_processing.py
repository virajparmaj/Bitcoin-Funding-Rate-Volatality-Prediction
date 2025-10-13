import pandas as pd
import numpy as np

from utilities.functions import (
    add_lag_features,
    add_technical_indicators,
    add_interaction_terms,
    remove_outliers,
    rescale_series,
    winsorize_series
)

# ===========================================
# Data Loading and Preprocessing
# ===========================================

def load_data(filepath, do_preview=True):
    try:
        df = pd.read_csv(filepath)
        if do_preview:
            print("Data loaded. Preview:")
            print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df, handle_timestamps=True):
    df = df.copy()

    if handle_timestamps and 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
            df.sort_values('timestamp', inplace=True)
        except Exception as e:
            print(f"Error during timestamp processing: {e}")
    elif handle_timestamps:
        print("'timestamp' column is missing. Skipping timestamp processing.")

    # Replace infinities with NaN first, to handle them consistently
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Forward fill so each missing value is replaced by the last valid one
    # If forward fill fails, fallback to 0 or leave it.
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)
    return df

# ===========================================
# Feature Engineering
# ===========================================

def create_features(df):
    """
    Create additional features and the target variable.
    """
    # Time-based and cyclical features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Create a future funding rate column
    df['future_funding_rate'] = df['funding_rate'].shift(-1)

    # Binary direction
    df['direction'] = (df['future_funding_rate'] > df['funding_rate']).astype(int)

    keep_future_rate=False
        
    if not keep_future_rate:
        # By default, you remove 'future_funding_rate' if you're only doing Model 1
        df.drop(columns=['future_funding_rate'], inplace=True)

    return df

# ===========================================
# Processing Pipeline
# ===========================================

def process_pipeline(filepath, rescale=False, scaling_factor=1e6,
                     handle_outliers=False, winsorize=False, winsorize_limits=(0.05, 0.05),
                     fill_method='ffill', keep_future_rate=False, drop_all_na=True):
    """
    Complete processing pipeline for loading, preprocessing, and feature creation.
    Includes options for rescaling and outlier removal.
    """
    try:
        print("Loading the dataset...")
        df = load_data(filepath)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print(df.head())  # Preview the initial data
    except Exception as e:
        print(f"Error during dataset loading: {e}")
        return None

    # Drop non-numeric columns that aren't needed for modeling
    # Adjust this list to remove any columns like 'exchange', 'symbol', 'local_timestamp', 'funding_timestamp' etc.
    columns_to_drop = ['exchange', 'symbol']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

    # Optionally winsorize the funding_rate column
    if winsorize:
        print(f"Winsorizing 'funding_rate' with limits {winsorize_limits}...")
        df['funding_rate'] = winsorize_series(df['funding_rate'], limits=winsorize_limits)
        print(f"Winsorization completed. 'funding_rate' head:\n{df['funding_rate'].head()}")

    # List of pipeline steps
    pipeline_steps = [
        ("Preprocessing the data", preprocess_data),
        ("Rescaling 'funding_rate'", lambda df: df.assign(funding_rate=rescale_series(df['funding_rate'], scaling_factor)) if rescale else df),
        ("Removing outliers from 'funding_rate'", lambda df: df.assign(funding_rate=remove_outliers(df['funding_rate'])) if handle_outliers else df),
        ("Adding lag features", add_lag_features),
        ("Adding technical indicators", lambda df: add_technical_indicators(df, ma_window=3, vol_window=5)),
        ("Adding interaction terms", add_interaction_terms),
        ("Creating features and target variable",
        lambda d: create_features(d, keep_future_rate=keep_future_rate)),
    ]
    
    for step_name, step_function in pipeline_steps:
        try:
            print(f"{step_name}...")
            df = step_function(df)
            print(f"{step_name} completed. Shape: {df.shape}")
            print(df.head())  # Optional: Preview data after each step
        except Exception as e:
            print(f"Error during {step_name.lower()}: {e}")
            return None

    if drop_all_na:
        # Drop only rows missing crucial columns
        essential_cols = ['funding_rate', 'direction']  # or whatever you need
        df.dropna(subset=essential_cols, inplace=True)
        print(f"Dropped rows with NaN in essential columns. Remaining shape: {df.shape}")
    else:
        print("Skipping global dropna(). Some columns may still have NaN.")
    
    return df
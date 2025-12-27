"""
Data import utilities for normalizing Tardis data.
"""

import gzip
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import (
    END_DATE,
    EXCHANGE,
    INSTRUMENT,
    NORMALIZED_DATA_DIR,
    RAW_DATA_DIR,
    START_DATE,
    TARDIS_TYPE,
)
from utilities.logger import get_logger, log_data_info, log_function_call

logger = get_logger(__name__)


@log_function_call
def normalize_tardis_data(
    exchange: str,
    tardis_type: str,
    start_date: str,
    end_date: str,
    instrument: str,
    raw_data_dir: Path,
    normalized_data_dir: Path,
) -> pd.DataFrame:
    """
    Normalize Tardis data by resampling to hourly frequency and combining daily files.
    
    Args:
        exchange: Exchange name
        tardis_type: Type of Tardis data
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        instrument: Trading instrument symbol
        raw_data_dir: Directory containing raw data files
        normalized_data_dir: Directory to save normalized data
        
    Returns:
        Combined and normalized DataFrame
    """
    logger.info(f"Starting data normalization for {exchange} {instrument}")
    
    # Convert start and end dates to datetime objects
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    logger.info(f"Processing data from {start_date} to {end_date}")

    # File pattern for raw data
    raw_file_pattern = raw_data_dir / f"{exchange}_{tardis_type}_{instrument}_{{date}}.csv.gz"
    # File pattern for normalized data
    normalized_file = normalized_data_dir / f"{exchange}_{instrument}_normalized.csv"

    dfs: List[pd.DataFrame] = []
    
    # Generate a list of dates within the given range
    date_list = [
        start_date_dt + timedelta(days=x) 
        for x in range((end_date_dt - start_date_dt).days)
    ]
    
    logger.info(f"Processing {len(date_list)} days of data")
    
    processed_count = 0
    skipped_count = 0
    missing_count = 0
    
    for date in date_list:
        date_str = date.strftime('%Y-%m-%d')
        filename = Path(str(raw_file_pattern).format(date=date_str))
        
        if filename.exists():
            try:
                with gzip.open(filename, 'rt') as f:
                    df = pd.read_csv(f)

                # Convert 'timestamp' to datetime and set it as the index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
                df = df.set_index('timestamp')

                # Resample to hourly frequency and take the last available entry
                df_hourly = df.resample('H').last()

                # Check if there are 24 data points; if not, skip the date
                if len(df_hourly) == 24:
                    dfs.append(df_hourly)
                    processed_count += 1
                    logger.debug(f"[PROCESSED] {date_str}")
                else:
                    skipped_count += 1
                    logger.warning(f"Skipping incomplete data for date: {date_str} "
                                f"(found {len(df_hourly)} hourly points, expected 24)")
                    
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                skipped_count += 1
        else:
            missing_count += 1
            logger.warning(f"File not found: {filename}")

    logger.info(f"Processing complete: {processed_count} files processed, "
                f"{skipped_count} files skipped, {missing_count} files missing")

    # Combine all DataFrames if there are any, or return an empty DataFrame
    if dfs:
        try:
            combined_df = pd.concat(dfs, ignore_index=False)
            
            # Log information about the combined dataset
            log_data_info(combined_df, "normalized_data", logger)
            
            # Ensure the output directory exists
            normalized_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the normalized data
            combined_df.to_csv(normalized_file)
            logger.info(f"Normalized data saved to {normalized_file}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error combining DataFrames: {str(e)}")
            return pd.DataFrame()
    else:
        logger.error("No valid data found in the given date range.")
        return pd.DataFrame()


if __name__ == "__main__":
    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    result_df = normalize_tardis_data(
        exchange=EXCHANGE,
        tardis_type=TARDIS_TYPE,
        start_date=START_DATE,
        end_date=END_DATE,
        instrument=INSTRUMENT,
        raw_data_dir=RAW_DATA_DIR,
        normalized_data_dir=NORMALIZED_DATA_DIR
    )
    
    if not result_df.empty:
        logger.info("Data normalization completed successfully")
    else:
        logger.error("Data normalization failed")

if __name__ == "__main__":
    normalize_tardis_data(
        exchange=EXCHANGE,
        tardis_type=TARDIS_TYPE,
        start_date=START_DATE,
        end_date=END_DATE,
        instrument=INSTRUMENT,
        raw_data_dir=RAW_DATA_DIR,
        normalized_data_dir=NORMALIZED_DATA_DIR
    )
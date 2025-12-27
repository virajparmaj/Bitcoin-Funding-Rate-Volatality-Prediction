"""
Data pull utilities for downloading data from Tardis API.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from tardis_dev import datasets

from config import (
    END_DATE,
    EXCHANGE,
    INSTRUMENT,
    RAW_DATA_DIR,
    START_DATE,
    TARDIS_TYPE,
)


def get_api_key() -> str:
    """
    Retrieve Tardis API key from environment variables.
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not found in environment variables
    """
    api_key = os.environ.get('TARDIS_API_KEY')
    if not api_key:
        raise ValueError("TARDIS_API_KEY not found in environment variables")
    return api_key


def download_tardis_data(
    exchange: str,
    start_date: str,
    end_date: str,
    tardis_sym: str,
    instrument: str,
    save_dir: Path,
) -> None:
    """
    Download data from Tardis API for the specified date range.
    
    Args:
        exchange: Exchange name
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        tardis_sym: Tardis data type symbol
        instrument: Trading instrument symbol
        save_dir: Directory to save downloaded data
    """
    api_key = get_api_key()

    cur_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Ensure save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    download_count = 0
    skip_count = 0

    while cur_date < end_date_dt:
        cur_str = cur_date.strftime("%Y-%m-%d")
        tom_str = (cur_date + timedelta(days=1)).strftime("%Y-%m-%d")

        filename = save_dir / f"{exchange}_{tardis_sym}_{instrument}_{cur_str}.csv.gz"
        
        if filename.exists():
            print(f"File already exists: {filename}")
            skip_count += 1
            cur_date += timedelta(days=1)
            continue

        try:
            # Download data using the Tardis API
            datasets.download(
                exchange=exchange,
                data_types=[tardis_sym],
                from_date=cur_str,
                to_date=tom_str,
                symbols=[instrument],
                api_key=api_key,
                download_dir=save_dir
            )
            
            print(f"[DOWNLOADED] {cur_str}")
            download_count += 1
            
        except Exception as e:
            print(f"Error downloading data for {cur_str}: {str(e)}")
        
        cur_date += timedelta(days=1)

    print(f"Download complete: {download_count} files downloaded, {skip_count} files skipped")


if __name__ == "__main__":
    # Ensure raw data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    download_tardis_data(
        exchange=EXCHANGE,
        start_date=START_DATE,
        end_date=END_DATE,
        tardis_sym=TARDIS_TYPE,
        instrument=INSTRUMENT,
        save_dir=RAW_DATA_DIR
    )

if __name__ == "__main__":
    download_tardis_data(
        exchange=EXCHANGE,
        start_date=START_DATE,
        end_date=END_DATE,
        tardis_sym=TARDIS_TYPE,
        instrument=INSTRUMENT,
        save_dir=RAW_DATA_DIR
    )
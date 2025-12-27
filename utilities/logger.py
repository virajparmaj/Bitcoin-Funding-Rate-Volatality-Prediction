"""
Logging utilities for the Bitcoin Funding Rate Volatility Prediction project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from config import (
    LOG_FILE,
    ERROR_LOG_FILE,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOGS_DIR
)


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = LOG_LEVEL,
    format_string: str = LOG_FORMAT,
    date_format: str = LOG_DATE_FORMAT
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Log message format
        date_format: Date format for log messages
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string, date_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = LOG_FILE
    
    # Ensure log directory exists
    log_file.parent.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def setup_error_logger(
    name: str,
    error_file: Optional[Path] = None,
    level: int = logging.ERROR,
    format_string: str = LOG_FORMAT,
    date_format: str = LOG_DATE_FORMAT
) -> logging.Logger:
    """
    Set up a dedicated error logger.
    
    Args:
        name: Logger name
        error_file: Path to error log file (optional)
        level: Logging level (typically ERROR)
        format_string: Log message format
        date_format: Date format for log messages
        
    Returns:
        Configured error logger instance
    """
    if error_file is None:
        error_file = ERROR_LOG_FILE
    
    return setup_logger(
        name=f"{name}_error",
        log_file=error_file,
        level=level,
        format_string=format_string,
        date_format=date_format
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    
    return wrapper


def log_data_info(df, name: str, logger: Optional[logging.Logger] = None):
    """
    Log information about a DataFrame.
    
    Args:
        df: DataFrame to log info about
        name: Name/identifier for the DataFrame
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"DataFrame '{name}' info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Data types:\n{df.dtypes}")
    logger.info(f"  Missing values: {df.isnull().sum().sum()}")
    
    if df.empty:
        logger.warning(f"DataFrame '{name}' is empty!")
    else:
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def log_model_performance(metrics: dict, model_name: str, logger: Optional[logging.Logger] = None):
    """
    Log model performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
        model_name: Name of the model
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"Model Performance - {model_name}:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


# Initialize default loggers
main_logger = setup_logger("bitcoin_funding_rate")
error_logger = setup_error_logger("bitcoin_funding_rate")
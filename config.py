import os
import logging
from pathlib import Path

# ===========================================
# Directory Configurations
# ===========================================

# Base directory of the project
BASE_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw_data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed_data'
NORMALIZED_DATA_DIR = DATA_DIR / 'normalized_datasets'

# Path to the main dataset
BINANCE_BTC_PERP_CSV = NORMALIZED_DATA_DIR / 'binance_btc_perp.csv'

# Models directory
MODELS_DIR = BASE_DIR / 'models'
SAVED_MODELS_DIR = MODELS_DIR / 'saved_models'

# Model file paths
MODEL1_RF_PATH = SAVED_MODELS_DIR / 'model1_RF.pkl'
SCALER1_RF_PATH = SAVED_MODELS_DIR / 'scaler1_RF.pkl'
MODEL1_LR_PATH = SAVED_MODELS_DIR / 'model1_LR.pkl'
SCALER1_LR_PATH = SAVED_MODELS_DIR / 'scaler1_LR.pkl'
MODEL2_GARCH_PATH = SAVED_MODELS_DIR / 'model2_GARCH.pkl'
MODEL3_RFR_PATH = SAVED_MODELS_DIR / 'model3_RFR.pkl'
MODEL3_SARIMAX_PATH = SAVED_MODELS_DIR / 'model3_SARIMAX.pkl'

# Instrument specifications
TARDIS_TYPE = 'derivative_ticker'
INSTRUMENT = 'btcusdt'
START_DATE = '2020-01-01'
END_DATE = '2024-10-18'
EXCHANGE = 'binance-futures'

# Results directory
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
PREDICTIONS_RFR_CSV = RESULTS_DIR / 'predictions_RFR.csv'
PREDICTIONS_SARIMAX_CSV = RESULTS_DIR / 'predictions_SARIMAX.csv'

# ===========================================
# Logging Configuration
# ===========================================

# Logging directory and files
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / 'bitcoin_funding_rate.log'
ERROR_LOG_FILE = LOGS_DIR / 'errors.log'

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ===========================================
# General Configurations
# ===========================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Data processing configurations
DEFAULT_SCALING_FACTOR = 1e6
DEFAULT_WINSORIZE_LIMITS = (0.05, 0.05)
DEFAULT_Z_SCORE_THRESHOLD = 3

# Model configurations
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
DEFAULT_SMOTE_STRATEGY = 1.0
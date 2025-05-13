"""
feature_engineer.py

This module provides functions to perform feature engineering on stock market data stored as parquet files.
It supports batch processing of all files in the RAW_DIR as well as single-file processing for a specified stock ID.
"""
import os, sys
import pandas as pd
import logging
import traceback
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RAW_DIR, PROCESSED_DIR, LOG_DIR, CHART_DATA_DIR
from features.volume_features import add_volume_features
from features.technical_indicators import add_technical_indicators
from features.price_features import add_price_features

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHART_DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Logger configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_name = f"feature_engineer_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_name), encoding="utf-8")
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
# Console handler: prints to stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def feature_engineer_single(stock_id: str) -> None:
    """
    Process one specific stock by *stock_id* (e.g. "2330").
    Wether filename is exist is checked by main.py.
    """

    filename = f"{stock_id}.parquet"
    input_path  = os.path.join(RAW_DIR, filename)
    output_path = os.path.join(PROCESSED_DIR, filename)
    chart_path  = os.path.join(CHART_DATA_DIR, filename)

    logger.info(f'Processing: {filename}')

    df = pd.read_parquet(input_path)

    # Make sure have the necessary fields
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    missing_columns = required_columns - set(df.columns)
    if not required_columns.issubset(df.columns):
        logger.warning(f'{filename} is missing required fields: {missing_columns}, skipped.')
        return

    try:
        # ==== START: Processing ====
        df = add_volume_features(df)
        df = add_technical_indicators(df)
        df = add_price_features(df)
        # ==== END: Processing ======

        df.to_parquet(output_path, index=False, compression="snappy")
        df.to_parquet(chart_path, index=False, compression="snappy")
        logger.info(f'Saved to: {output_path}')
        logger.info(f'Saved to: {chart_path}')

    except Exception as e:
        logger.error(f'{filename} processing failed: {e}')
        logger.error(traceback.format_exc())


def feature_engineer():
    """
    Run feature engineering pipeline.

    This function reads raw parquet files from the input directory, 
    performs feature engineering (including volume features, technical indicators, 
    and price-related features), and saves the processed data to the output directory.

    Logs processing progress, missing required fields, and errors during feature generation.
    """

    for filename in os.listdir(RAW_DIR):
        if not filename.endswith('.parquet'):
            continue

        stock_id, _ = os.path.splitext(filename)
        feature_engineer_single(stock_id)


if __name__ == "__main__":
    feature_engineer()

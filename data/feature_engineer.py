import os
import pandas as pd
import logging
import traceback
from datetime import datetime

from config import RAW_DIR, PROCESSED_DIR, LOG_DIR, CHART_DATA_DIR
from features.volume_features import add_volume_features
from features.technical_indicators import add_technical_indicators
from features.price_features import add_price_features

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHART_DATA_DIR, exist_ok=True)

# ==== Setup logger for feature_engineer module ====
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_name = f"feature_engineer_{datetime.now().strftime('%Y%m%d')}.log"

fh = logging.FileHandler(os.path.join(LOG_DIR, log_name))
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
# ==== Setup logger for feature_engineer module ====

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
            continue

        try:
            # ==== START: Processing ====
            df = add_volume_features(df)
            df = add_technical_indicators(df)
            df = add_price_features(df)
            # ==== END: Processing ======

            df.to_parquet(output_path, index=False)
            df.to_parquet(chart_path, index=False)
            logger.info(f'Saved to: {output_path}')
            logger.info(f'Saved to: {chart_path}')

        except Exception as e:
            logger.error(f'{filename} processing failed: {e}')
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    feature_engineer()

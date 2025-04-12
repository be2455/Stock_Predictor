import os
import pandas as pd
import logging
import traceback
from datetime import datetime

from features.volume_features import add_volume_features
from features.technical_indicators import add_technical_indicators
from features.price_features import add_price_features

INPUT_DIR  = 'data/raw'
OUTPUT_DIR = 'data/processed'
LOG_DIR    = 'data/log'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_name = f"feature_engineer_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    filename=os.path.join(LOG_DIR, log_name),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def feature_engineer():
    """
    Run feature engineering pipeline.

    This function reads raw parquet files from the input directory, 
    performs feature engineering (including volume features, technical indicators, 
    and price-related features), and saves the processed data to the output directory.

    Logs processing progress, missing required fields, and errors during feature generation.
    """

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith('.parquet'):
            continue

        input_path  = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        logging.info(f'Processing: {filename}')

        df = pd.read_parquet(input_path)

        # Make sure have the necessary fields
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing_columns = required_columns - set(df.columns)
        if not required_columns.issubset(df.columns):
            logging.warning(f'{filename} is missing required fields: {missing_columns}, skipped.')
            continue

        try:
            # ==== START: Processing ====
            df = add_volume_features(df)
            df = add_technical_indicators(df)
            df = add_price_features(df)
            # ==== END: Processing ======

            df.to_parquet(output_path, index=False)
            logging.info(f'Saved to: {output_path}')

        except Exception as e:
            logging.error(f'{filename} processing failed: {e}')
            logging.error(traceback.format_exc())

if __name__ == "__main__":
    feature_engineer()

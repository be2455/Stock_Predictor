import os, sys
import pandas as pd
import logging
import traceback
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RAW_DIR, PROCESSED_DIR, LOG_DIR, CHART_DATA_DIR, MARGIN_DIR
from features.volume_features import add_volume_features
from features.technical_indicators import add_technical_indicators
from features.price_features import add_price_features
from features.margin_features import add_margin_features

# ---------------------------------------------------------------------
# Ensure required directories exist
# ---------------------------------------------------------------------
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHART_DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Logger setup (idempotent)
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:  # avoid adding multiple handlers in interactive sessions
    logger.setLevel(logging.INFO)
    log_name = f"feature_engineer_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(os.path.join(LOG_DIR, log_name))
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(fh)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _validate_columns(df: pd.DataFrame, filename: str) -> bool:
    """Return True if all required columns are present, else log and return False."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.warning("%s missing required fields: %s, skipped.", filename, missing)
        return False
    return True


def _run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering steps in order."""
    # ==== START: Processing ====
    df = add_volume_features(df)
    df = add_technical_indicators(df)
    df = add_price_features(df)
    df = add_margin_features(df)
    # ==== END: Processing ======
    return df

# ---------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------
def feature_engineer(
    raw_dir: str | Path = RAW_DIR,
) -> None:
    """Process every *.parquet file in *raw_dir* and write results to *processed_dir* & *chart_dir*."""

    raw_dir = Path(raw_dir)
    for raw_path in raw_dir.glob("*.parquet"):
        filename = raw_path.name
        logger.info("Processing: %s", filename)
        try:
            df = pd.read_parquet(raw_path)
            if not _validate_columns(df, filename):
                continue
            df = _run_pipeline(df)

            # Save processed Parquet
            processed_path = Path(PROCESSED_DIR) / filename
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(processed_path, index=False)
            logger.info("Saved to: %s", processed_path)

            # Save chart data Parquet
            chart_path = Path(CHART_DATA_DIR) / filename
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(chart_path, index=False)
            logger.info("Saved to: %s", chart_path)

        except Exception as e:
            logger.error("%s processing failed: %s", filename, e)
            logger.error(traceback.format_exc())


# ---------------------------------------------------------------------
# Single‑stock processing
# ---------------------------------------------------------------------
def process_single_stock(stock_id: str) -> None:
    """Feature-engineer just one stock (e.g. stock_id='2330')."""
    raw_path = Path(RAW_DIR) / f"{stock_id}.parquet"
    margin_path = Path(MARGIN_DIR) / f"{stock_id}.parquet"
    if not raw_path.exists():
        logger.error("Raw file %s not found.", raw_path)
        raise FileNotFoundError(raw_path)
    if not margin_path.exists():
        logger.error("Margin file %s not found.", margin_path)
        raise FileNotFoundError(margin_path)

    df = pd.read_parquet(raw_path)
    if not _validate_columns(df, raw_path.name):
        return
    
    df_margin = pd.read_parquet(margin_path)
    df_margin = df_margin.rename(columns={'資料日期': 'date'})

    df_margin['date'] = pd.to_datetime(df_margin['date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(df_margin, on='date', how='left')

    df = _run_pipeline(df)

    # Output to processed and chart_data
    processed_path = Path(PROCESSED_DIR) / f"{stock_id}.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)

    chart_path = Path(CHART_DATA_DIR) / f"{stock_id}.csv"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(chart_path, index=False)

    logger.info("Finished feature engineering for %s", stock_id)

# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument("--stock", help="Specify a single stock code (eg: 2330)")
    return parser.parse_args()

if __name__ == "__main__":

    args = _parse_args()
    if args.stock:
        process_single_stock(args.stock)
    else:
        feature_engineer()

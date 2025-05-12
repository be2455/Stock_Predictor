from FinMind.data import DataLoader
import pandas as pd
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, sys
import logging
from datetime import datetime

from config import STOCK_LIST_PATH, RAW_DIR, LOG_DIR

RETRY_TIMES   = 3
SLEEP_SECONDS = 1
THREAD_NUM    = 5

load_dotenv()
API_Token = os.getenv('FINMIND_API_TOKEN')

api = DataLoader()
api.login_by_token(api_token=API_Token)

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==== Setup logger for fetch_stock_data module ====
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_name = f"fetch_stock_{datetime.now().strftime('%Y%m%d')}.log"

fh = logging.FileHandler(os.path.join(LOG_DIR, log_name), encoding="utf-8")
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
fh.setFormatter(formatter)
# Console handler: prints to stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(console_handler)
# ==== Setup logger for fetch_stock_data module ====

def fetch_stock(stock_id: str, start_date: str) -> None:
    """
    Download single stock data and store it in parquet.
    """
    output_file = os.path.join(RAW_DIR, f'{stock_id}.parquet')
    old_df: pd.DataFrame = pd.DataFrame()

    # ── The file already exists → check the latest date ────────────────────────────────
    if os.path.exists(output_file):
        try:
            old_df = pd.read_parquet(output_file)
            old_df["date"] = pd.to_datetime(old_df["date"]) 
            last_date = old_df["date"].max().normalize()
        except Exception as e:
            logger.warning(
                f"{stock_id} failed to read existing parquet ({e}), will re-fetch from {start_date}."
            )
            last_date, old_df = None, pd.DataFrame()

        # Determine whether the information is the latest
        if last_date is not None:
            today = pd.Timestamp.today().normalize()
            if last_date >= today:
                logger.info(f"{stock_id} has been updated to {last_date.date()}, skipping crawling")
                return
            else:
                # Move the start date back one day to avoid duplication
                start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info(
                    f"{stock_id} existing data to {last_date.date()}, "
                    f"The information will be supplemented starting from {start_date}."
                )

    # ── Start (or continue) capturing data ────────────────────────────────────
    for retry in range(RETRY_TIMES):
        try:
            new_df = api.taiwan_stock_daily(
                stock_id=stock_id,
                start_date=start_date,
                end_date=pd.Timestamp.today().strftime('%Y-%m-%d')
            )
            break
        except Exception as e:
            logger.warning(f'{stock_id} fetch failed {retry+1}/{RETRY_TIMES} times: {e}')
            time.sleep(SLEEP_SECONDS)
    else:
        logger.error(f"{stock_id} failed to fetch more than {RETRY_TIMES} times.")
        return

    if new_df.empty:
        logger.warning(f'{stock_id} No information.')
        return

    new_df = new_df.rename(columns={
        'Trading_Volume': 'Volume',
        'max': 'High',
        'min': 'Low',
        'open': 'Open',
        'close': 'Close'
    })

    df = pd.concat([old_df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset="date", keep="last")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # If the file already exists, append it; otherwise write it directly
    write_mode = "appended" if os.path.exists(output_file) else "overwritten"
    df.to_parquet(output_file, index=False, compression="snappy")
    logger.info(f"{stock_id} data has been {write_mode} to {output_file}.")
    return

def run_fetch():
    """
    Read stock_list.txt and fetch multiple stocks in parallel.
    """
    with open(STOCK_LIST_PATH, 'r') as f:
        stock_list = [line.strip().split('#')[0].split(',') for line in f]

    with ThreadPoolExecutor(max_workers=THREAD_NUM) as executor:
        futures = [
            executor.submit(fetch_stock, stock_id, start_date)
            for stock_id, start_date in stock_list
        ]

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    run_fetch()

from FinMind.data import DataLoader
import pandas as pd
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
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

fh = logging.FileHandler(os.path.join(LOG_DIR, log_name))
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
# ==== Setup logger for fetch_stock_data module ====

def fetch_stock(stock_id, start_date):
    """
    Download single stock data and store it in parquet.
    """
    output_file = os.path.join(RAW_DIR, f'{stock_id}.parquet')

    if os.path.exists(output_file):
        logger.info(f'{stock_id} already exists, skip')
        return

    for retry in range(RETRY_TIMES):
        try:
            df = api.taiwan_stock_daily(
                stock_id=stock_id,
                start_date=start_date,
                end_date=pd.Timestamp.today().strftime('%Y-%m-%d')
            )
        except Exception as e:
            logger.warning(f'{stock_id} fetch failed {retry+1}/{RETRY_TIMES} times: {e}')
            time.sleep(SLEEP_SECONDS)
            continue

        if df.empty:
            logger.warning(f'{stock_id} No information.')
            return

        df = df.rename(columns={
            'Trading_Volume': 'Volume',
            'max': 'High',
            'min': 'Low',
            'open': 'Open',
            'close': 'Close'
        })
        df.to_parquet(output_file)
        logger.info(f'{stock_id} archive completed.')
        return

    logger.error(f'{stock_id} fetch failed, retried {RETRY_TIMES} times failed')


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

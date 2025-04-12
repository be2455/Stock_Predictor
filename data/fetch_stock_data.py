from FinMind.data import DataLoader
import pandas as pd
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging
from datetime import datetime

DATA_PATH     = 'raw'
LOG_PATH      = 'log'
RETRY_TIMES   = 3
SLEEP_SECONDS = 1
THREAD_NUM    = 5

load_dotenv()
API_Token = os.getenv('FINMIND_API_TOKEN')

api = DataLoader()
api.login_by_token(api_token=API_Token)

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

log_name = f"fetch_stock_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    filename=os.path.join(LOG_PATH, log_name),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def fetch_stock(stock_id, start_date):
    """
    Download single stock data and store it in parquet.
    """
    output_file = os.path.join(DATA_PATH, f'{stock_id}.parquet')

    if os.path.exists(output_file):
        logging.info(f'{stock_id} already exists, skip')
        return

    for retry in range(RETRY_TIMES):
        try:
            df = api.taiwan_stock_daily(
                stock_id=stock_id,
                start_date=start_date,
                end_date=pd.Timestamp.today().strftime('%Y-%m-%d')
            )
        except Exception as e:
            logging.warning(f'{stock_id} fetch failed {retry+1}/{RETRY_TIMES} times: {e}')
            time.sleep(SLEEP_SECONDS)
            continue

        if df.empty:
            logging.warning(f'{stock_id} No information.')
            return

        df = df.rename(columns={
            'Trading_Volume': 'Volume',
            'max': 'High',
            'min': 'Low',
            'open': 'Open',
            'close': 'Close'
        })
        df.to_parquet(output_file)
        logging.info(f'{stock_id} archive completed.')
        return

    logging.error(f'{stock_id} fetch failed, retried {RETRY_TIMES} times failed')


def run_fetch(stock_list_file='stock_list.txt'):
    """
    Read stock_list.txt and fetch multiple stocks in parallel.
    """
    with open(stock_list_file, 'r') as f:
        stock_list = [line.strip().split(',') for line in f.readlines()]

    with ThreadPoolExecutor(max_workers=THREAD_NUM) as executor:
        futures = [
            executor.submit(fetch_stock, stock_id, start_date)
            for stock_id, start_date in stock_list
        ]

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    run_fetch()

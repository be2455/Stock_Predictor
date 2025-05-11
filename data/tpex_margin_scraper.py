import sys, os, time, glob
import logging
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import date, timedelta, datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CHIPS_DIR, MARGIN_DIR, STOCK_TPEX_LIST_PATH

# -----------------------------------------------------------------------------
# LOGGER SETUP
# -----------------------------------------------------------------------------
# Only message content in log; file named e.g. "3105_margi.log"
FMT = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
LOG_DIR = os.path.join(CHIPS_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_stock_logger(stock_id: str) -> logging.Logger:
    """
    Return a dedicated logger for the given stock_id.  
    Creates a file handler writing to '{stock_id}_margi.log' and a console handler.
    """
    logger = logging.getLogger(stock_id)
    logger.setLevel(logging.INFO)

    if not logger.handlers:                       # Avoid adding handlers repeatedly
        # File handler: writes to e.g. '3105_margi.log'
        fpath = os.path.join(LOG_DIR, f"{stock_id}_margin.log")
        fh = logging.FileHandler(fpath, encoding="utf-8")
        fh.setFormatter(FMT)
        # Console handler: prints to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(FMT)

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger

# -----------------------------------------------------------------------------
# FUNCTION: fetch_margin_safari
# -----------------------------------------------------------------------------
def fetch_margin_safari(date_str: str, download_dir: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetch the margin trading CSV for a given ROC date string from TPEx via Safari + Selenium.  

    Args:
        date_str: ROC-formatted date 'YYY/MM/DD', e.g. '114/05/06'.
        download_dir: local folder where Safari downloads CSV files.
        logger: logger instance to record progress.

    Returns:
        DataFrame containing the full margin trading table for that date.
    """
    # Start Safari (after running safaridriver --enable)
    driver = webdriver.Safari()
    driver.get("https://www.tpex.org.tw/zh-tw/mainboard/trading/margin-trading/transactions.html")

    wait = WebDriverWait(driver, 10)
    # Wait for the date input to appear
    inp = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.date[name='date']")))

    # arguments[0].removeAttribute('readonly');
        # Remove the readonly attribute on the <input> element, allowing it to be written programmatically.
    # arguments[0].value = arguments[1];
        # Set the value of this input box to the date string we want.
    # arguments[0].dispatchEvent(new Event('change'));
        # Manually trigger a change event to tell the page "the value of this field has just been changed",
        # usually the underlying JavaScript will reload the table data for that day.
    driver.execute_script("""
        arguments[0].removeAttribute('readonly');
        arguments[0].value = arguments[1];
        arguments[0].dispatchEvent(new Event('change'));
    """, inp, date_str)
    # Close the datepicker pop-up
    inp.send_keys(Keys.ESCAPE)
    time.sleep(2)  # Wait for JS to reload the table

    # Click the "Download CSV file (UTF‑8)" button
    btn_csv_u8 = driver.find_element(By.CSS_SELECTOR, "button[data-format='csv-u8']")
    driver.execute_script("arguments[0].click();", btn_csv_u8)

    # Wait for the file to appear in the Downloads folder
    pattern = os.path.join(download_dir, "*.csv")
    end_time = time.time() + 15
    csv_file = None
    while time.time() < end_time:
        files = glob.glob(pattern)
        if files:
            candidate = max(files, key=os.path.getctime)
            if time.time() - os.path.getctime(candidate) < 15:
                csv_file = candidate
                break
        time.sleep(1)

    driver.quit()
    if not csv_file:
        raise RuntimeError("CSV download failed or timed out")

    # Read the CSV, skip the first two description lines
    df = pd.read_csv(csv_file, encoding="utf-8-sig", skiprows=2)
    df.columns = df.columns.str.strip().str.replace("\ufeff","")

    # # Save as Parquet (daily)
    # os.makedirs(MARGIN_DIR, exist_ok=True)
    # fname = f"margin_{date_str.replace('/','')}.parquet"
    # parquet_path = os.path.join(MARGIN_DIR, fname)
    # df.to_parquet(parquet_path, index=False)

    # Clean up the temporary CSV file
    try:
        os.remove(csv_file)
        logger.info("Deleted temporary CSV: %s", csv_file)
    except OSError as e:
        logger.error("Error deleting CSV %s: %s", csv_file, e)

    return df


# -----------------------------------------------------------------------------
# MAIN: download_margin_for_list (single-horizon version)
# -----------------------------------------------------------------------------
def download_margin_for_list(download_dir: str) -> None:
    """
    Read stock list file, then for each stock ID, fetch margin data from its start date to today.
    Results are appended to '{stock_id}.parquet' in MARGIN_DIR.

    Args:
        download_dir: folder where Selenium downloads CSVs.
    """
    today = pd.Timestamp(date.today())

    # Read the list of 'stock_id,start_date#comment'
    with open(STOCK_TPEX_LIST_PATH, 'r') as f:
        stock_list = [line.strip().split('#')[0].split(',') for line in f]

    for stock_id, start_date in stock_list:
        logger = get_stock_logger(stock_id)
        logger.info(f"Start fetching {stock_id} …")

        # Parse start date and enforce minimum of 2007-01-01
        year, month, day = start_date.split('-')
        start = datetime(int(year), int(month), int(day)).date()
        start = max(start, date(2007, 1, 1))  # Ensure the start date is no earlier than 2007‑01‑01

        # Load existing parquet if present
        out_path = os.path.join(MARGIN_DIR, f"{stock_id}.parquet")
        old_df = pd.DataFrame()
        current = pd.Timestamp(start)
        if os.path.exists(out_path):
            old_df = pd.read_parquet(out_path)
            if not old_df.empty:
                # Find the last day for which you have data
                last_date_str = old_df["資料日期"].max()
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
                # Next business day (pd.Timestamp + BDay)
                current = pd.Timestamp(last_date) + BDay()
                logger.info(
                    f"▲ {stock_id} already has data up to {last_date_str}. "
                    f"Will continue from {current.date()}"
                )

        dfs: List[pd.DataFrame] = []
        try:
            while current < today:
                # skip weekends
                if current.weekday() >= 5:
                    current += timedelta(days=1)
                    continue

                roc = f"{current.year-1911}/{current.month:02}/{current.day:02}"
                try:
                    daily_df = fetch_margin_safari(roc, download_dir, logger)
                    row = daily_df[daily_df["代號"] == stock_id]
                    if not row.empty:
                        row = row.copy()
                        row["資料日期"] = current.strftime("%Y-%m-%d")
                        dfs.append(row)
                        logger.info(f"✓ {current} data available")
                    else:
                        logger.info(f"- {current} no matching stock id")
                except RuntimeError as e:
                    logger.error(f"✘ {current} download failed: {e}")

                current += BDay()

        except KeyboardInterrupt:
            logger.info("⚠️  Detected Ctrl+C, saving progress and exiting early…")

        finally:
            if dfs:
                new_df = pd.concat([old_df] + dfs, ignore_index=True).drop_duplicates(subset=["資料日期"])
                os.makedirs(MARGIN_DIR, exist_ok=True)

                # Define a set of columns to exclude from numeric conversion
                exclude = {"代號", "名稱", '備註', "資料日期"}

                # Build a list of all columns whose dtype is object (strings)
                # but not in the exclude set—these are the columns to convert
                to_numeric_cols = [
                    c for c, dt in new_df.dtypes.items()
                    if dt == "object" and c not in exclude
                ]

                # For each column identified, 
                #   1. cast values to string (in case they are mixed types)
                #   2. remove any thousands‑separator commas
                #   3. convert the result to numeric, coercing invalid parses to NaN
                for c in to_numeric_cols:
                    new_df[c] = (
                        new_df[c].astype(str)
                                .str.replace(",", "", regex=False)
                                .pipe(pd.to_numeric, errors="coerce")
                    )

                new_df.to_parquet(out_path, index=False)
                logger.info(f"→  Saved to {out_path}")
            else:
                logger.warning("(No new data to write)")


if __name__ == "__main__":
    download_dir = os.path.expanduser("/Users/Ks/Downloads")
    download_margin_for_list(download_dir)

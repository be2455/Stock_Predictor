# -*- coding: utf-8 -*-
"""tpex_institution_scraper_with_types.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fetch daily *major institutional investors* (三大法人買賣明細) data from TPEx and
append it into a per-stock Parquet archive.

Key points
----------
* **Browser**   : Safari driven by Selenium WebDriver.
* **Encoding**  : CSV files are Big5-HK (`big5hkscs`).
* **Earliest**  : TPEx provides data starting 2007-04-20 (ROC 96/04/20).
* **Persistence** : Results are appended to `<stock_id>_institution.parquet`.
* **Logging**   : One log per stock (e.g. ``3105_institution.log``).
"""

from __future__ import annotations

import pandas as pd
import logging
import sys, os, time, glob
from pandas.tseries.offsets import BDay
from datetime import date, timedelta, datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List

# ---------------------------------------------------------------------------
# Project‑level paths
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CHIPS_DIR, MARGIN_DIR, STOCK_TPEX_LIST_PATH

# -----------------------------------------------------------------------------
# LOGGER SETUP (one logger per stock symbol)
# -----------------------------------------------------------------------------
# Only message content in log; file named e.g. "3105_institution.log"
FMT = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
LOG_DIR = os.path.join(CHIPS_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_stock_logger(stock_id: str) -> logging.Logger:  # noqa: D401
    """Return a dedicated *per‑stock* logger.

    The logger writes to ``<stock_id>_institution.log`` under ``LOG_DIR`` and
    simultaneously echoes to *stdout*.
    """
    logger = logging.getLogger(stock_id)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Avoid adding handlers repeatedly
        file_path = os.path.join(LOG_DIR, f"{stock_id}_institution.log")
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(FMT)
        # Console handler: prints to stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FMT)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# ---------------------------------------------------------------------------
# CORE SCRAPER: fetch_institution_safari
# ---------------------------------------------------------------------------
def fetch_institution_safari(date_str: str,
                            download_dir: str,
                            logger: logging.Logger) -> pd.DataFrame:
    """
    Download 'major institutional investors detail / day' CSV for a given ROC date.

    Args:
        date_str : ROC date, e.g. '114/05/08'
        download_dir : Safari download directory
        logger : Logger instance

    Returns:
        Pandas DataFrame of the full daily table.
    """
    # --- 1. Trigger CSV download via Selenium --------------------------------
    # after running safaridriver --enable
    driver = webdriver.Safari()
    driver.get(
        "https://www.tpex.org.tw/zh-tw/mainboard/trading/major-institutional/detail/day.html"
    )

    wait = WebDriverWait(driver, 10)
    inp = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input.date[name='date']"))
    )

    # unblock <input readonly>  then set value and fire change event
    driver.execute_script(
        """
        arguments[0].removeAttribute('readonly');
        arguments[0].value = arguments[1];
        arguments[0].dispatchEvent(new Event('change'));
        """,
        inp,
        date_str,
    )
    inp.send_keys(Keys.ESCAPE)  # close calendar pop‑up
    time.sleep(2)               # wait for table to refresh

    # click *Save CSV* button (Only one button)
    btn_csv = driver.find_element(By.CSS_SELECTOR, "button[data-format='csv']")
    driver.execute_script("arguments[0].click();", btn_csv)

    # --- 2. Wait for the file to land in download_dir ------------------------
    pattern = os.path.join(download_dir, "*.csv")
    deadline = time.time() + 15
    csv_path = None
    while time.time() < deadline:
        files = glob.glob(pattern)
        if files:
            newest = max(files, key=os.path.getctime)
            # ensure file is very recent (avoid picking old remnants)
            if time.time() - os.path.getctime(newest) < 15:
                csv_path = newest
                break
        time.sleep(1)

    driver.quit()
    if not csv_path:
        raise RuntimeError("CSV download failed or timed out (15s)")

    # --- 3. Load CSV – handle Big5/Big5‑HK encodings -------------------------
    for enc in ("utf-8-sig", "big5hkscs", "cp950", "big5"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, skiprows=1)
            break          # Jump out if reading is successful
        except UnicodeDecodeError:
            continue
    if df is None:
        # last resort – ignore undecodable bytes
        df = pd.read_csv(csv_path, encoding="big5hkscs", header=1, encoding_errors="ignore")

    df.columns = df.columns.str.strip().str.replace("\ufeff", "")

    # --- 4. Clean‑up ---------------------------------------------------------
    try:
        os.remove(csv_path)
        logger.info("Deleted tmp CSV: %s", csv_path)
    except OSError as e:
        logger.error("Error deleting CSV %s: %s", csv_path, e)

    return df

# ---------------------------------------------------------------------------
# BATCH DRIVER: download_institution_for_list
# ---------------------------------------------------------------------------
def download_institution_for_list(download_dir: str) -> None:
    """Iterate through STOCK_TPEX_LIST_PATH and update each stock's Parquet.

    The txt list must be in the form::

        3105,2011-12-13#comment  ← start date is in Gregorian calendar
        00679B,2020-08-04        ← multiple lines are allowed

    For each stock, the function:
    1. Reads existing ``<id>_institution.parquet`` if present.
    2. Crawls forward from the day *after* the latest stored date (or the
       configured start date) up to *yesterday*.
    3. Appends new rows and writes back the Parquet.
    """
    today = pd.Timestamp(date.today())

    with open(STOCK_TPEX_LIST_PATH, "r") as f:
        stock_list: List[List[str]] = [line.strip().split("#")[0].split(",") for line in f]

    for stock_id, start_date in stock_list:
        logger = get_stock_logger(f"{stock_id}_institution")
        logger.info(f"Start fetching institutional detail for {stock_id} …")

        # ---- compute crawl start -----------------------------------------
        y, m, d = start_date.split("-")
        start = datetime(int(y), int(m), int(d)).date()
        start = max(start, date(2007, 4, 20))   # TPEx earliest

        out_path = os.path.join(MARGIN_DIR, f"{stock_id}_institution.parquet")
        old_df = pd.read_parquet(out_path) if os.path.exists(out_path) else pd.DataFrame()

        current = pd.Timestamp(start)
        if not old_df.empty:
            last_str = old_df["資料日期"].max()
            last_day = datetime.strptime(last_str, "%Y-%m-%d").date()
            current = pd.Timestamp(last_day) + BDay()
            logger.info(
                f"▲ {stock_id} already has data up to {last_str}. "
                f"Will continue from {current.date()}"
            )

        dfs: List[pd.DataFrame] = []

        try:
            while current < today:
                if current.weekday() >= 5:   # weekend
                    current += timedelta(days=1)
                    continue

                roc = f"{current.year-1911}/{current.month:02}/{current.day:02}"
                try:
                    full_df = fetch_institution_safari(roc, download_dir, logger)
                    row = full_df[full_df["代號"] == stock_id]
                    if not row.empty:
                        row = row.copy()
                        row["資料日期"] = current.strftime("%Y-%m-%d")
                        dfs.append(row)
                        logger.info(f"✓ {current} data ok")
                    else:
                        logger.info(f"- {current} no id")
                except RuntimeError as e:
                    logger.error(f"✘ {current} failed: {e}")

                current += BDay()

        except KeyboardInterrupt:
            logger.info("⚠️  Detected Ctrl+C, saving progress and exiting early…")

        finally:
            if dfs:
                new_df = pd.concat([old_df] + dfs, ignore_index=True).drop_duplicates(subset=["資料日期"])
                os.makedirs(MARGIN_DIR, exist_ok=True)

                # Define a set of columns to exclude from numeric conversion
                exclude = {"代號", "名稱", "資料日期"}

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
                logger.info("→ saved %s", out_path)
            else:
                logger.warning("no new data")

# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    download_dir = os.path.expanduser("/Users/Ks/Downloads")
    download_institution_for_list(download_dir)

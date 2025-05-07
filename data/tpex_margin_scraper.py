import sys, os, time, glob
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import date, timedelta, datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MARGIN_DIR, STOCK_TPEX_LIST_PATH

def fetch_margin_safari(date_str, download_dir):
    # Start Safari (after running safaridriver --enable)
    driver = webdriver.Safari()
    driver.get("https://www.tpex.org.tw/zh-tw/mainboard/trading/margin-trading/transactions.html")

    wait = WebDriverWait(driver, 10)
    # Waiting for data date input box
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

    # Read CSV
    df = pd.read_csv(csv_file, encoding="utf-8-sig", skiprows=2)
    df.columns = df.columns.str.strip().str.replace("\ufeff","")

    # # Save as Parquet (daily)
    # os.makedirs(MARGIN_DIR, exist_ok=True)
    # fname = f"margin_{date_str.replace('/','')}.parquet"
    # parquet_path = os.path.join(MARGIN_DIR, fname)
    # df.to_parquet(parquet_path, index=False)

    # Delete the original CSV
    try:
        os.remove(csv_file)
        print("Deleted temporary CSV:", csv_file)
    except OSError as e:
        print("Error deleting CSV:", e)

    return df
    # return df[df["代號"] == "3105"]


def download_margin_for_list(download_dir):
    today = pd.Timestamp(date.today())

    with open(STOCK_TPEX_LIST_PATH, 'r') as f:
        stock_list = [line.strip().split('#')[0].split(',') for line in f]

    for stock_id, start_date in stock_list:
        year, month, day = start_date.split('-')
        start = datetime(int(year), int(month), int(day)).date()
        start = max(start, date(2007, 1, 1))  # Ensure the start date is no earlier than 2007‑01‑01

        dfs = []
        current = pd.Timestamp(start)
        while current < today:
            # skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            roc = f"{current.year-1911}/{current.month:02}/{current.day:02}"
            try:
                daily_df = fetch_margin_safari(roc, download_dir)
                row = daily_df[daily_df["代號"] == stock_id]
                if not row.empty:
                    row = row.copy()
                    row["資料日期"] = current.strftime("%Y-%m-%d")
                    dfs.append(row)
                    print(f"✓ {current} 有資料")
                else:
                    print(f"- {current} 無該代號")
            except RuntimeError as e:
                print(f"× {current} 下載失敗: {e}")

            current += BDay()

        if dfs:
            all_df = pd.concat(dfs, ignore_index=True)
            os.makedirs(MARGIN_DIR, exist_ok=True)
            out_path = os.path.join(MARGIN_DIR, f"{stock_id}.parquet")
            all_df.to_parquet(out_path, index=False)
            print(f"→ 已存 {out_path}")
        else:
            print("整段期間都沒有資料")



if __name__ == "__main__":
    download_dir = os.path.expanduser("/Users/Ks/Downloads")
    download_margin_for_list(download_dir)

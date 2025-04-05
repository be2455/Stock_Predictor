import yfinance as yf
import pandas as pd
import os

output_dir = "data/raw"
output_path = os.path.join(output_dir, "TSMC_2330TW_90d.csv")

TSMC = yf.Ticker("2330.TW")
data = TSMC.history(period="90d", interval="1d")
data = data.drop(columns=["Dividends", "Stock Splits"])

data.to_csv(output_path)

if data.empty:
    print("Warning: Download failed or no data, please check the ticker or network connection.")
else :
    print(f"TSMC stock price information has been saved to : {output_path}")
"""multi_horizon_training.py
================================
Train & evaluate regression / classification models for **multiple forecast horizons**
(3, 5, 10, 20 days by default) using the existing pipelines.
"""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd
from typing import List, Optional

# === Project‑level imports ====================================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, STOCK_LIST_PATH
from data.fetch_stock_data import fetch_stock
from data.feature_engineer import feature_engineer_single
from train_utils import *

# ──────────────────── utilities ────────────────────
def get_start_date(stock_id: str, list_path: str = STOCK_LIST_PATH) -> Optional[str]:
    """Return the start-date string in stock_list.txt for given stock_id."""
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            core = line.split("#")[0].strip()
            if not core:
                continue
            sid, start = map(str.strip, core.split(","))
            if sid == stock_id:
                return start
    return None

# -----------------------------------------------------------------------------
# MAIN (multi‑horizon version)
# -----------------------------------------------------------------------------
def main() -> None:
    """End-to-end workflow for **multi-horizon** training / evaluation."""

    # ---------------- CLI ----------------
    parser = argparse.ArgumentParser(description="Train/evaluate multi-horizon stock models")
    parser.add_argument("stock_id", help="Stock Symbol (e.g. 2330)")
    parser.add_argument("--horizons", "-H", type=int, nargs="*", default=[5, 10, 20, 60],
                        help="Forecast horizons in days (default: 5 10 20 60)")
    parser.add_argument("--model", choices=["gbdt", "nn"], default="gbdt",
                        help="'gbdt' = HistGradientBoosting, 'nn' = PyTorch neural net")
    args = parser.parse_args()

    # --------------- Data acquisition  & feature engineering -------------
    start_date = get_start_date(args.stock_id)
    if start_date is None:
        # logger.warning("Stock %s not found in %s", args.stock_id, STOCK_LIST_PATH)
        sys.exit(f"[Error] {args.stock_id} not found in {STOCK_LIST_PATH}")

    print(f"📥  Fetching {args.stock_id} starting {start_date} ...")
    fetch_stock(args.stock_id, start_date)
    feature_engineer_single(args.stock_id)

    # ---- dynamic import *after* args are known ----
    if args.model == "nn":
        from pipelines.regression_pipeline import build_nn_pipeline \
            as build_regression_pipeline
        from pipelines.classification_pipeline import build_nn_pipeline \
            as build_classification_pipeline
        print("⚙️  Using Neural-Net pipelines")
    else:
        from pipelines.regression_pipeline import build_regression_pipeline
        from pipelines.classification_pipeline import build_classification_pipeline
        print("⚙️  Using Gradient-Boosting pipelines")

    # ---------- Load engineered data ----------
    input_path = os.path.join(PROCESSED_DIR, f"{args.stock_id}.parquet")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_parquet(input_path)

    # time period characteristics ----------------------------------------
    Date = pd.to_datetime(df['date'])
    df['DayOfWeek'] = Date.dt.dayofweek      # 0=Mon
    df['Month']     = Date.dt.month
    df['IsHolidayEve'] = (Date.shift(-1) - Date).dt.days > 3  # 簡單版 IsHolidayEve：若下一個交易日間隔超過3天視為長週末

    # Feature / raw column split -----------------------------------------
    raw_cols = [
        'date', 'stock_id', 'Volume', 'Trading_money', 
        'Open', 'High', 'Low', 'Close', 'spread', 'Trading_turnover'
    ]
    feature_cols: List[str] = sorted(set(df.columns) - set(raw_cols))

    # ——————— Convert price–volume quadrant features to categorical ———————
    for period in [5, 10, 20, 60]:
        columns_to_convert = [
            f'price_up_volume_up_{period}d',
            f'price_up_volume_down_{period}d',
            f'price_down_volume_up_{period}d',
            f'price_down_volume_down_{period}d',
            f'price_volume_quadrant_{period}d'
        ]

        for column in columns_to_convert:
            df[column] = df[column].astype('category')

    # ——————— Convert technical-indicator cross signals to categorical ———————
    df['RSI_cross'] = df['RSI_cross'].astype('category')
    df['macd_cross_signal'] = df['macd_cross_signal'].astype('category')
    df['MACD_cross_zero'] = df['MACD_cross_zero'].astype('category')

    # ------- Convert time period to categorical -------------------
    df['DayOfWeek'] = df['DayOfWeek'].astype('category')
    df['Month'] = df['Month'].astype('category')
    df['IsHolidayEve'] = df['IsHolidayEve'].astype('category')

    # TODO:
    # df['Margin Balance Δ Sign'] = np.sign(df['Margin Balance (shares) Δ'])
    # df['Short Balance Δ Sign'] = np.sign(df['Short Balance (shares) Δ'])

    # ---------- Iterate horizons ----------
    for horizon in args.horizons:
        print(f"\n================  Horizon {horizon}-day  ================")

        # Create horizon‑specific targets
        Y = add_return_and_target(df["Close"], horizon=horizon)

        # Align X with target indices (because of shifting)
        X = df.loc[Y.index, feature_cols]

        # Targets
        y_reg = Y[f"return_fwd_{horizon}"]
        y_cls = Y[f"target_up_{horizon}"]

        # Pipelines (new instance per horizon)
        reg_pipeline = build_regression_pipeline(X)
        cls_pipeline = build_classification_pipeline(X)

        # Train & evaluate
        train_and_evaluate(X, y_reg, y_cls, reg_pipeline, cls_pipeline)

if __name__ == "__main__":
    main()

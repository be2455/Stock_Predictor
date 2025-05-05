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
from typing import List

# === Project‑level imports ====================================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR
from data.fetch_stock_data import run_fetch
from data.feature_engineer import feature_engineer
from train_utils import *

# -----------------------------------------------------------------------------
# MAIN (multi‑horizon version)
# -----------------------------------------------------------------------------
def main() -> None:
    """End-to-end workflow for **multi-horizon** training / evaluation."""

    # Data acquisition & feature engineering
    run_fetch()
    feature_engineer()

    # ---------------- CLI ----------------
    parser = argparse.ArgumentParser(description="Train/evaluate multi-horizon stock models")
    parser.add_argument("stock_id", help="Stock Symbol (e.g. 2330)")
    parser.add_argument("--horizons", "-H", type=int, nargs="*", default=[5, 10, 20, 60],
                        help="Forecast horizons in days (default: 5 10 20 60)")
    parser.add_argument("--model", choices=["gbdt", "nn"], default="gbdt",
                        help="'gbdt' = HistGradientBoosting, 'nn' = PyTorch neural net")
    args = parser.parse_args()

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

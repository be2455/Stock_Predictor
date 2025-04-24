import pandas as pd
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASELINE_RESULT_DIR, PROCESSED_DIR
from classification_baseline import baseline_direction
from regression_baseline import baseline_return
from sklearn.metrics import mean_squared_error, accuracy_score

os.makedirs(BASELINE_RESULT_DIR, exist_ok=True)

def compute_targets(
        df: pd.DataFrame,
        close_col: str = 'Close',
        horizon: int = 10
) -> pd.DataFrame:
    """ Add the rate of return and the RISE/FALL label."

    Args:
        df (pd.DataFrame): The input must include a closing price column.
        close_col (str): The name of the closing price field, default is 'Close'.
        horizon (int): The number of days to predict (the return rate in a few days). The default value is 10.

    Returns:
        pd.DataFrame: Returns a DataFrame with the new return rate and category label fields.
    """

    return_col = f'return_{horizon}d'
    label_col = f'label_{horizon}d_up'

    df[return_col] = (df[close_col].shift(-horizon) - df[close_col]) / df[close_col]
    df[label_col] = (df[return_col] > 0).astype(int)
    return df


def evaluate_baselines_multi_horizon(df: pd.DataFrame, horizons: list[int] = [3, 5, 10, 20]) -> pd.DataFrame:
    """
    Evaluate baseline models for multiple horizons (both regression and classification tasks)

    Args:
        df (pd.DataFrame): The stock price data (must include 'Close' column)
        horizons (list[int]): A list of horizons to evaluate (in days)

    Returns:
        pd.DataFrame: A DataFrame with residuals, columns like '3d_modelA', '5d_modelB', etc.
    """
    all_residuals = {}

    for horizon in horizons:
        print(f"\n=== Horizon: {horizon} days ===")

        df = compute_targets(df, horizon=horizon)

        return_col = f'return_{horizon}d'
        label_col = f'label_{horizon}d_up'

        # === Regression baseline ===
        print("→ Regression (Return Prediction)")
        return_preds = baseline_return(df)
        for name, pred in return_preds.items():
            pred_series = pd.Series(pred, index=df.index)
            mask = ~df[return_col].isna() & ~pd.Series(pred).isna()
            residuals = df.loc[mask, return_col] - pred_series[mask]

            mse = mean_squared_error(df.loc[mask, return_col], pd.Series(pred)[mask])
            print(f"   {name:>12}: MSE = {mse:.6f}")

            col_name = f"{horizon}d_{name}"
            all_residuals[col_name] = residuals

        # === Classification baseline ===
        print("→ Classification (RISE/FALL Prediction)")
        direction_preds = baseline_direction(df)
        for name, pred in direction_preds.items():
            mask = ~df[label_col].isna() & ~pd.Series(pred).isna()
            acc = accuracy_score(df.loc[mask, label_col], pd.Series(pred)[mask])
            print(f"   {name:>12}: Accuracy = {acc:.4f}")

    residuals_df = pd.DataFrame(all_residuals)
    return residuals_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stock_id", help="Stock Symbol")
    args = parser.parse_args()

    filename    = f"{args.stock_id}.parquet"
    input_path  = os.path.join(PROCESSED_DIR, filename)
    output_path = os.path.join(BASELINE_RESULT_DIR, filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_parquet(input_path)
    residuals_df = evaluate_baselines_multi_horizon(df, horizons=[3, 5, 10, 20])
    residuals_df.to_parquet(output_path)

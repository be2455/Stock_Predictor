import pandas as pd
import argparse
import os
from classification_baseline import baseline_direction
from regression_baseline import baseline_return
from sklearn.metrics import mean_squared_error, accuracy_score

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


def evaluate_baselines_multi_horizon(df: pd.DataFrame, horizons: list[int] = [3, 5, 10, 20]) -> None:
    """
    Evaluate baseline models for multiple horizons (both regression and classification tasks)

    Args:
        df (pd.DataFrame): The stock price data (must include 'Close' column)
        horizons (list[int]): A list of horizons to evaluate (in days)

    Returns:
        None: Print baseline MSE and Accuracy for each horizon
    """
    for horizon in horizons:
        print(f"\n=== Horizon: {horizon} days ===")

        df = compute_targets(df, horizon=horizon)

        return_col = f'return_{horizon}d'
        label_col = f'label_{horizon}d_up'

        # === Regression baseline ===
        print("→ Regression (Return Prediction)")
        return_preds = baseline_return(df)
        for name, pred in return_preds.items():
            mask = ~df[return_col].isna() & ~pd.Series(pred).isna()
            mse = mean_squared_error(df.loc[mask, return_col], pd.Series(pred)[mask])
            print(f"   {name:>12}: MSE = {mse:.6f}")

        # === Classification baseline ===
        print("→ Classification (RISE/FALL Prediction)")
        direction_preds = baseline_direction(df)
        for name, pred in direction_preds.items():
            mask = ~df[label_col].isna() & ~pd.Series(pred).isna()
            acc = accuracy_score(df.loc[mask, label_col], pd.Series(pred)[mask])
            print(f"   {name:>12}: Accuracy = {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stock_id", help="Stock Symbol")
    args = parser.parse_args()

    file_path = f"data/processed/{args.stock_id}.parquet"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_parquet(file_path)
    evaluate_baselines_multi_horizon(df, horizons=[3, 5, 10, 20])

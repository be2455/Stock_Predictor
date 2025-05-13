import pandas as pd

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Volume Moving Average (5,10,20,60)
    - Volume ratio (trading volume / average volume)
    - Volume fluctuation (standard deviation)
    """

    Volume = df['Volume']

    vol_windows = [5, 10, 20, 60]
    for window in vol_windows:
        vol_ma = Volume.rolling(window=window).mean()
        vol_std = Volume.rolling(window=window).std()

        df[f'Vol MA {window}D'] = vol_ma               # Moving Average
        df[f'Vol Ratio {window}D'] = Volume / vol_ma   # Volume ratio
        df[f'Vol STD {window}D'] = vol_std  # Volume volatility (standard deviation of volume)
        df[f'Vol Z{window}'] = (Volume - vol_ma) / vol_std

    return df

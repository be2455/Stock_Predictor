import pandas as pd

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Volume Moving Average (5,10,20)
    - Volume ratio (trading volume / average volume)
    - Volume fluctuation (standard deviation)
    """

    Volume = df['Volume']

    vol_windows = [5, 10, 20]
    for window in vol_windows:
        vol_ma = Volume.rolling(window=window).mean()

        df[f'Vol_ma_{window}'] = vol_ma               # Moving Average
        df[f'Vol_ratio_{window}'] = Volume / vol_ma   # Volume ratio
        df[f'Vol_std_{window}'] = Volume.rolling(window=window).std()  # Volume volatility (standard deviation of volume)

    return df

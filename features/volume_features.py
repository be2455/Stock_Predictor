import pandas as pd

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Volume Moving Average (5,10,20)
    - Volume ratio (trading volume / average volume)
    - Volume fluctuation (standard deviation)
    """

    Volume = df['Volume']

    # Moving Average
    Vol_ma_5  = Volume.rolling(window=5).mean()
    Vol_ma_10 = Volume.rolling(window=10).mean()
    Vol_ma_20 = Volume.rolling(window=20).mean()

    df['Vol_ma_5']  = Vol_ma_5
    df['Vol_ma_10'] = Vol_ma_10
    df['Vol_ma_20'] = Vol_ma_20

    # Volume ratio
    df['Vol_ratio_5']  = Volume / Vol_ma_5
    df['Vol_ratio_10'] = Volume / Vol_ma_10
    df['Vol_ratio_20'] = Volume / Vol_ma_20

    # Volume volatility (standard deviation of volume)
    df['Vol_std_5']  = Volume.rolling(window=5).std()
    df['Vol_std_10'] = Volume.rolling(window=10).std()

    return df

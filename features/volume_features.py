import pandas as pd

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Volume Moving Average (5,10,20)
    - Volume ratio (trading volume / average volume)
    - Volume fluctuation (standard deviation)
    """

    # Moving Average
    df['Vol_ma_5']  = df['Volume'].rolling(window=5).mean()
    df['Vol_ma_10'] = df['Volume'].rolling(window=10).mean()
    df['Vol_ma_20'] = df['Volume'].rolling(window=20).mean()

    # Volume ratio
    df['Vol_ratio_5']  = df['Volume'] / df['Vol_ma_5']
    df['Vol_ratio_10'] = df['Volume'] / df['Vol_ma_10']

    # Volume volatility (standard deviation of volume)
    df['Vol_std_5']  = df['Volume'].rolling(window=5).std()
    df['Vol_std_10'] = df['Volume'].rolling(window=10).std()

    return df

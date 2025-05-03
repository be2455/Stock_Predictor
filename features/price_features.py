import pandas as pd

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    Open   = df['Open']
    Close  = df['Close']
    High   = df['High']
    Low    = df['Low']
    Volume = df['Volume']
    Prev_Close = df['Close'].shift(1)

    Price_Features = {}

    # intraday
    Price_Features['daily_volatility'] = (High - Low) / Close
    Price_Features['gap']              = (Open - Prev_Close) / Prev_Close
    Price_Features['intraday_change']  = (Close - Open) / Open

    for period in [5, 10, 20, 60]:

        # ----- SMA simple moving average -----
        sma = Close.rolling(period).mean()
        df[f'SMA_{period}'] = sma
        df[f'SMA_ratio_{period}'] = Close / sma
        slope = sma - sma.shift(1)
        df[f'SMA_slope_{period}'] = slope
        df[f'SMA_slope_{period}_avg_3'] = slope.rolling(3).mean()
        df[f'SMA_slope_{period}_avg_5'] = slope.rolling(5).mean()

        # Calculate percentage change in closing price over different periods
        Price_Features[f'return_{period}d'] = Close.pct_change(periods=period)

        # Calculate rolling volatility (standard deviation of daily returns) over different periods
        # This reflects price fluctuation or risk level in the given time window
        # Skip 1d volatility
        if period > 1:
            Price_Features[f"Price_volatility_{period}d"] = Close.pct_change().rolling(window=period).std()

        # Generate price-volume quadrant features over multiple time windows
        price_change  = f'price_change_{period}d'
        volume_change = f'volume_change_{period}d'
        
        Price_Features[price_change]  = Close - Close.shift(period)
        Price_Features[volume_change] = Volume - Volume.shift(period)

        Price_Features[f'price_up_volume_up_{period}d'] = (
            (Price_Features[price_change] > 0) & (Price_Features[volume_change] > 0)
        ).astype(int)

        Price_Features[f'price_up_volume_down_{period}d'] = (
            (Price_Features[price_change] > 0) & (Price_Features[volume_change] <= 0)
        ).astype(int)

        Price_Features[f'price_down_volume_up_{period}d'] = (
            (Price_Features[price_change] < 0) & (Price_Features[volume_change] > 0)
        ).astype(int)

        Price_Features[f'price_down_volume_down_{period}d'] = (
            (Price_Features[price_change] < 0) & (Price_Features[volume_change] <= 0)
        ).astype(int)

        Price_Features[f'price_volume_quadrant_{period}d'] = (
            Price_Features[f'price_up_volume_up_{period}d'] * 1 +
            Price_Features[f'price_up_volume_down_{period}d'] * 2 +
            Price_Features[f'price_down_volume_up_{period}d'] * 3 +
            Price_Features[f'price_down_volume_down_{period}d'] * 4
        )

    df['SMA5_vs_SMA20']  = (df['SMA_5']  - df['SMA_20']) / df['SMA_20']
    df['SMA20_vs_SMA60'] = (df['SMA_20'] - df['SMA_60']) / df['SMA_60']

    df = pd.concat([df, pd.DataFrame(Price_Features)], axis=1)

    return df

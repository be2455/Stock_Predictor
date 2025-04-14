import pandas as pd

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    Close  = df['Close']
    High   = df['High']
    Low    = df['Low']
    Volume = df['Volume']

    Price_Features = {}

    # Daily price range normalized by close price (volatility)
    Price_Features['Price_range'] = (High - Low) / Close

    for period in [1, 3, 5, 14, 20, 60]:

        # Calculate percentage change in closing price over different periods
        Price_Features[f'pct_change_{period}d'] = Close.pct_change(periods=period)

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

    df = pd.concat([df, pd.DataFrame(Price_Features)], axis=1)

    return df

import pandas as pd
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators: RSI, MACD, KD, SMA, Bollinger Bands
    Assume that df already contains open/high/low/close/volume fields.
    """

    Close = df['Close']
    High  = df['High']
    Low   = df['Low']

    # ----- RSI -----
    RSI_14 = ta.momentum.RSIIndicator(close = Close, window = 14).rsi()
    df['RSI_14']      = RSI_14
    df['RSI_14_diff'] = RSI_14 - RSI_14.shift(1)

    # ----- MACD -----
    MACD      = ta.trend.MACD(close = Close)
    MACD_Diff = MACD.macd_diff()
    df['MACD']            = MACD.macd()
    df['MACD_signal']     = MACD.macd_signal()
    df['MACD_diff']       = MACD_Diff
    df['MACD_diff_delta'] = MACD_Diff.diff()

    df['MACD_cross_zero'] = 0
    df.loc[(df['MACD'].shift(1) < 0) & (df['MACD'] >= 0), 'MACD_cross_zero'] = 1   # travel upwards
    df.loc[(df['MACD'].shift(1) > 0) & (df['MACD'] <= 0), 'MACD_cross_zero'] = -1  # Cross down

    # ----- KD (Stochastic Oscillator) -----
    stoch = ta.momentum.StochasticOscillator(high = High, low = Low, close = Close)
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()

    # ----- SMA simple moving average -----
    SMA_5  = ta.trend.SMAIndicator(close = Close, window =  5).sma_indicator()
    SMA_10 = ta.trend.SMAIndicator(close = Close, window = 10).sma_indicator()
    SMA_20 = ta.trend.SMAIndicator(close = Close, window = 20).sma_indicator()
    SMA_60 = ta.trend.SMAIndicator(close = Close, window = 60).sma_indicator()

    df['SMA_5']  = SMA_5
    df['SMA_10'] = SMA_10
    df['SMA_20'] = SMA_20
    df['SMA_60'] = SMA_60

    df['SMA_ratio_5']  = Close / SMA_5
    df['SMA_ratio_10'] = Close / SMA_10
    df['SMA_ratio_20'] = Close / SMA_20
    df['SMA_ratio_60'] = Close / SMA_60

    SMA_slope_5  = SMA_5 - SMA_5.shift(1)
    SMA_slope_10 = SMA_10 - SMA_10.shift(1)
    SMA_slope_20 = SMA_20 - SMA_20.shift(1)
    SMA_slope_60 = SMA_60 - SMA_60.shift(1)

    df['SMA_slope_5']  = SMA_slope_5
    df['SMA_slope_10'] = SMA_slope_10
    df['SMA_slope_20'] = SMA_slope_20
    df['SMA_slope_60'] = SMA_slope_60

    # TODO: Determine the average of N days
    df['SMA_slope_20_avg_3'] = SMA_slope_20.rolling(3).mean()
    df['sma_slope_20_avg_5'] = SMA_slope_20.rolling(5).mean()

    # ----- EMA exponential moving average -----
    df['EMA_5']  = ta.trend.EMAIndicator(close = Close, window =  5).ema_indicator()
    df['EMA_10'] = ta.trend.EMAIndicator(close = Close, window = 10).ema_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(close = Close, window = 20).ema_indicator()
    df['EMA_60'] = ta.trend.EMAIndicator(close = Close, window = 60).ema_indicator()

    # ----- Bollinger Bands -----
    bollinger = ta.volatility.BollingerBands(close = Close, window = 20, window_dev = 2)
    df['bollinger_mavg']  = bollinger.bollinger_mavg()
    df['bollinger_upper'] = bollinger.bollinger_hband()
    df['bollinger_lower'] = bollinger.bollinger_lband()
    df['bollinger_width'] = df['bollinger_upper'] - df['bollinger_lower']

    df.bfill(inplace=True)  # Fill in missing values ​​(avoid training errors)

    return df
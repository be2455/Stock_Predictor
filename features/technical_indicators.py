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
    df['RSI_14'] = ta.momentum.RSIIndicator(close = Close, window = 14).rsi()

    # ----- MACD -----
    MACD = ta.trend.MACD(close = Close)
    df['MACD']        = MACD.macd()
    df['MACD_signal'] = MACD.macd_signal()
    df['MACD_diff']   = MACD.macd_diff()

    # ----- KD (Stochastic Oscillator) -----
    stoch = ta.momentum.StochasticOscillator(high = High, low = Low, close = Close)
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()

    # ----- SMA simple moving average -----
    df['SMA_5']  = ta.trend.SMAIndicator(close = Close, window =  5).sma_indicator()
    df['SMA_10'] = ta.trend.SMAIndicator(close = Close, window = 10).sma_indicator()
    df['SMA_20'] = ta.trend.SMAIndicator(close = Close, window = 20).sma_indicator()
    df['SMA_60'] = ta.trend.SMAIndicator(close = Close, window = 60).sma_indicator()

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
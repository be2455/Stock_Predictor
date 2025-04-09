import pandas as pd
import ta

# Parameters for MACD: (short_term, long_term, signal_line)
DIF_fast = 12
DIF_slow = 26
DEA = 9

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators: SMA, EMA, RSI, MACD, KD, Bollinger Bands
    Assume that df already contains open/high/low/close/volume fields.
    """

    Close = df['Close']
    High  = df['High']
    Low   = df['Low']

    sma_windows = [5, 10, 20, 60]
    for window in sma_windows:
        # ----- SMA simple moving average -----
        sma = ta.trend.SMAIndicator(close=Close, window=window).sma_indicator()
        df[f'SMA_{window}'] = sma
        df[f'SMA_ratio_{window}'] = Close / sma
        df[f'SMA_slope_{window}'] = sma - sma.shift(1)
        df[f'SMA_slope_{window}_avg_3'] = df[f'SMA_slope_{window}'].rolling(3).mean()
        df[f'SMA_slope_{window}_avg_5'] = df[f'SMA_slope_{window}'].rolling(5).mean()

        # ----- EMA exponential moving average -----
        df[f'EMA_{window}'] = ta.trend.EMAIndicator(close=Close, window=window).ema_indicator()

    # ----- RSI -----
    rsi_windows = [7, 14, 21]
    for window in rsi_windows:
        rsi_col  = f'RSI_{window}'
        diff_col = f'RSI_{window}_diff'

        df[rsi_col]  = ta.momentum.RSIIndicator(close=Close, window=window).rsi()
        df[diff_col] = df[rsi_col] - df[rsi_col].shift(1)

    df['RSI_7_21_spread'] = df['RSI_7'] - df['RSI_21']
    df['RSI_7_21_spread_prev'] = df['RSI_7_21_spread'].shift(1)

    df['RSI_cross'] = 0
    df.loc[(df['RSI_7_21_spread_prev'] < 0) & (df['RSI_7_21_spread'] > 0), 'RSI_cross'] = 1   # Yesterday RSI_7 was below RSI_21, today RSI_7 crossed above
    df.loc[(df['RSI_7_21_spread_prev'] > 0) & (df['RSI_7_21_spread'] < 0), 'RSI_cross'] = -1  # Yesterday RSI_7 was above RSI_21, today RSI_7 crossed below

    df.drop(columns=['RSI_7_21_spread_prev'], inplace=True)

    # ----- MACD -----
    MACD = ta.trend.MACD(
        close=Close,
        window_slow=DIF_slow,
        window_fast=DIF_fast,
        window_sign=DEA
    )
    MACD_Diff = MACD.macd_diff()

    df['DIF']  = MACD.macd()
    df['MACD'] = MACD.macd_signal()

    df['macd_cross_signal'] = 0
    # DIF crosses upwards through MACD
    df.loc[(df['DIF'].shift(1) < df['MACD'].shift(1)) & (df['DIF'] > df['MACD']), 'macd_cross_signal'] = 1
    # DIF falls below MACD
    df.loc[(df['DIF'].shift(1) > df['MACD'].shift(1)) & (df['DIF'] < df['MACD']), 'macd_cross_signal'] = -1

    df['Histogram']       = MACD_Diff
    df['Histogram_delta'] = MACD_Diff.diff()

    df['MACD_cross_zero'] = 0
    df.loc[(df['DIF'].shift(1) < 0) & (df['DIF'] >= 0), 'MACD_cross_zero'] = 1   # travel upwards
    df.loc[(df['DIF'].shift(1) > 0) & (df['DIF'] <= 0), 'MACD_cross_zero'] = -1  # Cross down

    macd_windows = [3, 5, 7]
    for window in macd_windows:
        df[f'macd_diff_{window}_mean'] = MACD_Diff.rolling(window).mean()
        df[f'macd_diff_{window}_std'] =  MACD_Diff.rolling(window).std()

    # ----- KD (Stochastic Oscillator) -----
    stoch = ta.momentum.StochasticOscillator(high = High, low = Low, close = Close)
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()

    # ----- Bollinger Bands -----
    bollinger = ta.volatility.BollingerBands(close = Close, window = 20, window_dev = 2)
    df['bollinger_mavg']  = bollinger.bollinger_mavg()
    df['bollinger_upper'] = bollinger.bollinger_hband()
    df['bollinger_lower'] = bollinger.bollinger_lband()
    df['bollinger_width'] = df['bollinger_upper'] - df['bollinger_lower']

    return df
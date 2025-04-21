import pandas as pd
import numpy as np
from typing import Dict
from numpy.typing import NDArray

def baseline_return(df: pd.DataFrame) -> Dict[str, NDArray[int]]:
    """ Baseline prediction for regression tasks """
    baseline: Dict[str, NDArray[int]] = {}

    # Predicted return rate of 0
    baseline['zero_return'] = np.zeros(len(df))
    # Calculate the return rate for the day (The percentage change of today's closing price compared to yesterday)
    df['return_today'] = df['Close'].pct_change()
    # Use the current day's return rate as the forecast value
    baseline['today_return'] = df['return_today']

    return baseline

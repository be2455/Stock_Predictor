import pandas as pd
import numpy as np
from typing import Dict
from numpy.typing import NDArray

def baseline_direction(df: pd.DataFrame) -> Dict[str, NDArray[int]]:
    """ Baseline prediction for Classification tasks """
    baseline: Dict[str, NDArray[int]] = {}

    # always predict 'up'
    baseline['always_up'] = np.ones(len(df))
    # random guess
    baseline['random'] = np.random.choice([0, 1], size=len(df))
    # The forecast (rise and fall) is same with the previous day.
    df['prev_label'] = (df['Close'] > df['Close'].shift(1)).astype(int)
    baseline['follow_prev'] = df['prev_label']

    return baseline

"""train_model.py
=================
Core utilities for target engineering and generic model training / evaluation.

* **add_return_and_target** - create forward-looking returns and binary
  direction labels for multiple horizons.
* **train_and_evaluate** - fits a regression and classification pipeline
  (already pre-constructed by caller) and prints / returns basic metrics.

This file is *model-agnostic*: any scikit-learn compatible `Pipeline` that
implements `.fit(X, y)` & `.predict(X)` can be dropped in.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Sequence, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import RegressorMixin, ClassifierMixin

__all__ = [
    "add_return_and_target",
    "train_and_evaluate",
]

###############################################################################
# Target engineering helpers
###############################################################################

def _ensure_iterable_horizons(
    horizon: int | None,
    horizons: Sequence[int] | None,
) -> List[int]:
    """Utility: resolve the *actual* list of horizons to use."""
    if horizon is not None and horizons is not None:
        raise ValueError("Specify either `horizon` or `horizons`, not both.")
    if horizon is not None:
        return [int(horizon)]
    if horizons is None:
        horizons = (5, 10, 20, 60)
    return list(horizons)

def add_return_and_target(
    close: pd.Series,
    *,
    horizon: int | None = None,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Generate forward returns & direction labels for one or many horizons.

    Parameters
    ----------
    close : pd.Series
        Series of closing prices ordered by time (index ascending).
    horizon : int, optional
        A single forecast horizon (e.g. 5). Mutually exclusive with
        `horizons`.
    horizons : sequence[int], optional
        Multiple forecast horizons. Ignored if `horizon` is given.

    Returns
    -------
    pd.DataFrame
        Columns: ``return_fwd_{h}``, ``target_up_{h}`` for each *h*.
        The last *max(horizons)* rows are dropped because they lack a
        forward price for target calculation.
    """
    used_horizons = _ensure_iterable_horizons(horizon, horizons)

    df = pd.DataFrame(index=close.index)
    max_h = max(used_horizons)

    for h in used_horizons:
        fwd_return = close.shift(-h) / close - 1.0
        df[f"return_fwd_{h}"] = fwd_return
        df[f"target_up_{h}"] = (fwd_return > 0).astype(int)

    # remove rows without full forward information (i.e. tail of series)
    df = df.iloc[:-max_h] if max_h > 0 else df
    df = df.dropna(how="any")
    return df

###############################################################################
# Training / evaluation helper
###############################################################################

def train_and_evaluate(
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_cls: pd.Series,
    reg_pipeline: RegressorMixin,
    cls_pipeline: ClassifierMixin,
    test_size: float = 0.2,
    *,
    shuffle: bool = False,
) -> Dict[str, float]:
    """Fit *both* pipelines and report their hold-out performance.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (no target columns).
    y_reg : pd.Series
        Regression target (continuous forward return).
    y_cls : pd.Series
        Classification target (0/1 direction label).
    reg_pipeline : RegressorMixin
        A scikit-learn regressor with `fit` / `predict`.
    cls_pipeline : ClassifierMixin
        A scikit-learn classifier with `fit` / `predict`.
    test_size : float, default 0.2
        Fraction of samples reserved for test set.
    shuffle : bool, default False
        Whether to shuffle samples when splitting.

    Returns
    -------
    dict
        ``{"mse": ..., "accuracy": ...}``
    """

    # consistent split (time‑series aware by default)
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=test_size, shuffle=shuffle
    )
    _, _, y_cls_train, y_cls_test = train_test_split(
        X, y_cls, test_size=test_size, shuffle=shuffle
    )
    
    # --- Train ---
    reg_pipeline.fit(X_train, y_reg_train)
    cls_pipeline.fit(X_train, y_cls_train)

    # --- Evaluate ---
    y_reg_pred = reg_pipeline.predict(X_test)
    y_cls_pred = cls_pipeline.predict(X_test)

    mse = mean_squared_error(y_reg_test, y_reg_pred)
    acc = accuracy_score(y_cls_test, y_cls_pred)

    print(f"Regressor MSE: {mse:.6f}")
    print(f"Classifier ACC: {acc:.4f}")

    return {"mse": mse, "accuracy": acc}

from __future__ import annotations

from typing import List, Sequence
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

def build_regression_pipeline(
    X: pd.DataFrame,
    cat_features: Sequence[str] | None = None
) -> Pipeline:
    """
    Build a sklearn pipeline that includes normalization of numerical features, 
    One-Hot encoding of categorical features, and a regression model.

    Args:
        X (pd.DataFrame): Feature data, including numerical and categorical features.

    Returns:
        Pipeline: sklearn's regression model pipeline.
    """
    if cat_features is None:
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_features: List[str] = (
        X.select_dtypes(include=['float64', 'int64']).columns.difference(cat_features).tolist()
    )

    num_features: List[str] = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=1000,
            l2_regularization=1.0,
            random_state=42,
        ))
    ])
    return reg_pipeline

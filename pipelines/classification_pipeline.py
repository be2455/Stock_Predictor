from __future__ import annotations

from typing import List, Sequence
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import ClassifierMixin


def build_classification_pipeline(
    X: pd.DataFrame,
    *,
    cat_features: Sequence[str] | None = None,
    model: ClassifierMixin | None = None,
) -> Pipeline:
    """
    Return a classification pipeline with:
      • Imputation  (num: median, cat: most_frequent)
      • Scaling     (StandardScaler) for numerical columns
      • One-Hot     encoding for categorical columns
      • A configurable classifier (default: HistGradientBoostingClassifier)

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature table.
    cat_features : list-like[str] | None
        Column names to treat as categorical. If None, auto-detect
        dtype == object/category.
    model : sklearn classifier | None
        Custom estimator. If None, uses HistGradientBoostingClassifier().

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if cat_features is None:
        cat_features: List[str] = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_features: List[str] = X.select_dtypes(include=['float64', 'int64']).columns.difference(cat_features).tolist()

    
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_features),
            ('cat', cat_pipe, cat_features)
        ],
        remainder='drop'
    )
    
    if model is None:
        model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )

    cls_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ],
        memory=None
    )

    return cls_pipeline

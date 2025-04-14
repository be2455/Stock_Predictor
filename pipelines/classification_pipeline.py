from typing import List
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier


def build_classification_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build an sklearn Pipeline for a classification task.

    This pipeline contains:
    - Standardization of numerical features (StandardScaler)
    - OneHot Encoding of Categorical Features (OneHotEncoder)
    - GradientBoostingClassifier as the final classification model

    Args:
    X : pd.DataFrame
        The original feature data table is used to determine the value and category fields.

    Returns:
    Pipeline
        sklearn pipeline that can be used for training and prediction.
    """
    num_features: List[str] = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_features: List[str] = [col for col in X.columns if 'quadrant' in col]
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    cls_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier())
    ])
    return cls_pipeline

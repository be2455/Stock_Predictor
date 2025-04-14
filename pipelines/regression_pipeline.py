from typing import List
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

def build_regression_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build a sklearn pipeline that includes normalization of numerical features, 
    One-Hot encoding of categorical features, and a regression model.

    Args:
        X (pd.DataFrame): Feature data, including numerical and categorical features.

    Returns:
        Pipeline: sklearn's regression model pipeline.
    """
    num_features: List[str] = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_features: List[str] = [col for col in X.columns if 'quadrant' in col]
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor())
    ])
    return reg_pipeline

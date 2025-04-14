import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import RegressorMixin, ClassifierMixin


def add_return_and_target(Close: pd.Series) -> pd.DataFrame:
    """
    Calculate future return and direction.

    Args:
        Close (pd.Series): Closing price series.

    Returns:
        pd.DataFrame: Data containing future returns and up/down labels.
    """
    df = pd.DataFrame(index=Close.index)

    horizons = [3, 5, 10, 20]
    for h in horizons:
        df[f'return_fwd_{h}'] = Close.shift(-h) / Close - 1
        df[f'target_up_{h}'] = (df[f'return_fwd_{h}'] > 0).astype(int)

    df.dropna(inplace=True)  # 20 lines will be delete
    return df


def train_and_evaluate(
    X: pd.DataFrame, 
    y_reg: pd.Series, y_cls: pd.Series, 
    reg_pipeline: RegressorMixin, 
    cls_pipeline: ClassifierMixin
) -> None:
    """
    Train regression and classification models and evaluate performance on test data.

    Args:
        X (pd.DataFrame): feature data
        y_reg (pd.Series): regression target value
        y_cls (pd.Series): categorical target value
        reg_pipeline (RegressorMixin): Regression model Pipeline
        cls_pipeline (ClassifierMixin): Classification Model Pipeline
    """

    # Split data: 80% training / 20% testing, keep the time series unshuffled (shuffle=False)
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, shuffle=False
    )
    # The target of the same segmentation category
    _, _, y_cls_train, y_cls_test = train_test_split(
        X, y_cls, test_size=0.2, shuffle=False
    )
    
    reg_pipeline.fit(X_train, y_reg_train)
    cls_pipeline.fit(X_train, y_cls_train)
    
    print("Regressor MSE:", mean_squared_error(y_reg_test, reg_pipeline.predict(X_test)))
    print("Classifier ACC:", accuracy_score(y_cls_test, cls_pipeline.predict(X_test)))

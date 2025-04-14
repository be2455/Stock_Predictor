from sklearn.impute import SimpleImputer

from data.fetch_stock_data import run_fetch
from data.feature_engineer import feature_engineer
from pipelines.regression_pipeline import build_regression_pipeline
from pipelines.classification_pipeline import build_classification_pipeline
from model.train_model import add_return_and_target, train_and_evaluate

import pandas as pd

def main():
    run_fetch()
    feature_engineer()

    df = pd.read_parquet('data/processed/3105.parquet')

    # Get feature list
    raw_cols = [
        'date', 'stock_id', 'Volume', 'Trading_money', 
        'Open', 'High', 'Low', 'Close', 'spread', 'Trading_turnover'
    ]
    feature_cols = list(set(df.columns) - set(raw_cols))

    Y = add_return_and_target(df['Close'])
    X = df.loc[Y.index, feature_cols]  # Align the X and Y indices

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    y_reg = Y['return_fwd_5']
    y_cls = Y['target_up_5']
    reg_pipeline = build_regression_pipeline(X)
    cls_pipeline = build_classification_pipeline(X)

    train_and_evaluate(X, y_reg, y_cls, reg_pipeline, cls_pipeline)

if __name__ == "__main__":
    main()
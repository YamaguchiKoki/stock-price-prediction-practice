import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def regression_metrics(estimator, X, y):
    """回帰モデルの評価指標を計算する"""
    y_pred = estimator.predict(X)
    return pd.DataFrame([{
        "RMSE": mean_squared_error(y, y_pred),
        "R2": r2_score(y, y_pred)
    }])

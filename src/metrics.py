import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

def regression_metrics(estimator, X, y):
    """回帰モデルの評価指標を計算する"""
    y_pred = estimator.predict(X)
    return pd.DataFrame([{
        "RMSE": mean_squared_error(y, y_pred),
        "R2": r2_score(y, y_pred)
    }])

def classification_metrics(estimator, X, y, threshold=0.5):
    """分類モデルの評価指標を計算する"""
    # predictメソッドは確率を返すように実装する
    y_pred_proba = estimator.predict(X)
    y_pred = (y_pred_proba >= threshold).astype(int)

    return pd.DataFrame([{
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred)
    }])

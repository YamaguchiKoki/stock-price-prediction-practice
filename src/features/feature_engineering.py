import pandas as pd

def create_features(
    df: pd.DataFrame,
    lags : list[int] = [1, 2, 3],
    windows: list[int] = [3, 5, 10],
) -> pd.DataFrame:
    """
    日足データから有用な特徴量を生成

    Parameters:
        df (pd.DataFrame): [Date, Open, High, Low, Close, Up] を含む DataFrame
        lags (list[int]): 過去の値を参照するためのラグ数
        windows (list[int]): 移動平均や標準偏差の計算に使うウィンドウ

    Returns:
        pd.DataFrame: 特徴量付きデータフレーム
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df["dayofweek"] = df.index.dayofweek

    for col in ["Open", "High", "Low", "Close"]:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag) # lag行前の価格
            df[f"{col}_diff_{lag}"] = df[col] - df[col].shift(lag) # lag行前の価格との差

    # 高値と安値の差
    df["high_low_diff"] = df["High"] - df["Low"]

    # 前日の終値との差
    df["close_diff"] = df["Close"].diff()
    df["close_pct_change"] = df["Close"].pct_change()

    for window in windows:
        df[f"close_ma_{window}"] = df["Close"].rolling(window).mean() # 移動平均
        df[f"close_std_{window}"] = df["Close"].rolling(window).std() # ボラテリティ
        df[f"close_max_{window}"] = df["Close"].rolling(window).max()
        df[f"close_min_{window}"] = df["Close"].rolling(window).min()
        df[f"close_range_{window}"] = df[f"close_max_{window}"] - df[f"close_min_{window}"]

    # トレンドストリーク（連続上昇 or 下降日数）
    df["close_up"] = (df["Close"].diff() > 0).astype(int)
    df["streak"] = df["close_up"].groupby((df["close_up"] != df["close_up"].shift()).cumsum()).cumcount() + 1
    df["streak"] *= df["close_up"].replace(0, -1)

    df.dropna(inplace=True)
    return df


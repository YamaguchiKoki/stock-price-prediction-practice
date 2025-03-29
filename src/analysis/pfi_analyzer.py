from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd


class PFIAnalyzer:
    def __init__(self, model, X_val, y_val):
        """
        Parameters:
            model: XGBoost モデル
            X_val 検証用特徴量
            y_val  検証用ラベル
        """
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.importances_ = None

    def compute(self, n_repeats: int = 30, scoring: str = "accuracy", random_state: int = 42) -> pd.DataFrame:
        """PFI を計算し、結果を DataFrame で返す"""
        result = permutation_importance(
            self.model,
            self.X_val,
            self.y_val,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1
        )
        self.importances_ = pd.DataFrame({
            "feature": self.X_val.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }).sort_values("importance_mean", ascending=False)
        return self.importances_

    def plot(self, top_n: int = 20):
        """上位 top_n の重要特徴量をプロット"""
        if self.importances_ is None:
            raise ValueError("PFI has not been computed yet. Call `compute()` first.")
        df = self.importances_.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(df["feature"], df["importance_mean"], xerr=df["importance_std"], color="skyblue")
        plt.gca().invert_yaxis()
        plt.title("Permutation Feature Importance")
        plt.xlabel("Importance (mean decrease in score)")
        plt.tight_layout()
        plt.show()

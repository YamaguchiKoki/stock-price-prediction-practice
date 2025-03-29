import xgboost as xgb
from src.models.model import Model
from pathlib import Path

class ModelXGB(Model):
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        validation = va_x is not None and va_y is not None
        dtrain = xgb.DMatrix(tr_x, label=tr_y)

        if validation:
            dvalid = xgb.DMatrix(va_x, label=va_y)

        # ハイパーパラメータの取得と除去
        params = dict(self.params)
        print(params)
        num_round = params.pop('num_round', 1000)
        early_stopping_rounds = params.pop('early_stopping_rounds', None)

        if validation and early_stopping_rounds is not None:
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_round,
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
        else:
            watchlist = [(dtrain, 'train')]
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_round,
                evals=watchlist,
                verbose_eval=False
            )

    def predict(self, te_x):
        dtest = xgb.DMatrix(te_x)
        if hasattr(self.model, "best_iteration"):
            return self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))
        return self.model.predict(dtest)

    def save_model(self, path: Path = Path("xgb_model.json")) -> None:
        """モデルの保存を行う"""
        self.model.save_model(str(path))

    def load_model(self, path: Path = Path("xgb_model.json")) -> None:
        """モデルの読み込みを行う"""
        self.model = xgb.Booster()
        self.model.load_model(str(path))

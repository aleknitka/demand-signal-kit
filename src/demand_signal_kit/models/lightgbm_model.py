import polars as pl
import lightgbm as lgb
import numpy as np
from demand_signal_kit.models.base import BaseForecaster
from demand_signal_kit.models.registry import register_model


@register_model("lightgbm")
class LightGBMForecaster(BaseForecaster):
    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.05, **kwargs):
        self.params = {"n_estimators": n_estimators, "learning_rate": learning_rate, **kwargs}
        self._models: dict[str, object] = {}
        self._feature_columns: list[str] = []

    def fit(
        self,
        train_df: pl.DataFrame,
        date_column: str,
        target_column: str,
        product_column: str,
        feature_columns: list[str],
    ) -> None:
        self._feature_columns = feature_columns
        products = train_df[product_column].unique().to_list()

        for pid in products:
            pdf = train_df.filter(pl.col(product_column) == pid)
            X = pdf.select(feature_columns).to_numpy()
            y = pdf[target_column].to_numpy()

            model = lgb.LGBMRegressor(**self.params)
            model.fit(X, y)
            self._models[pid] = model

    def predict(
        self,
        future_df: pl.DataFrame,
        date_column: str,
        product_column: str,
        feature_columns: list[str],
    ) -> pl.DataFrame:
        results = []
        products = future_df[product_column].unique().to_list()

        for pid in products:
            if pid not in self._models:
                continue
            model = self._models[pid]
            pdf = future_df.filter(pl.col(product_column) == pid)
            X = pdf.select(feature_columns).to_numpy()
            preds = model.predict(X)
            result = pdf.select([date_column, product_column]).with_columns(
                pl.Series("forecast", np.clip(preds, 0, None))
            )
            results.append(result)

        if not results:
            return pl.DataFrame()
        return pl.concat(results)

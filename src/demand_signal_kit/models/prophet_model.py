import polars as pl
import pandas as pd
from demand_signal_kit.models.base import BaseForecaster
from demand_signal_kit.models.registry import register_model


@register_model("prophet")
class ProphetForecaster(BaseForecaster):
    def __init__(self, **kwargs):
        self.params = kwargs
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
        from prophet import Prophet

        self._feature_columns = feature_columns
        products = train_df[product_column].unique().to_list()

        for pid in products:
            pdf = (
                train_df.filter(pl.col(product_column) == pid)
                .select([date_column, target_column] + feature_columns)
                .rename({date_column: "ds", target_column: "y"})
                .to_pandas()
            )
            model = Prophet(**self.params)
            for feat in feature_columns:
                model.add_regressor(feat)
            model.fit(pdf)
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
            pdf = (
                future_df.filter(pl.col(product_column) == pid)
                .select([date_column] + feature_columns)
                .rename({date_column: "ds"})
                .to_pandas()
            )
            forecast = model.predict(pdf)
            pdf_result = forecast[["ds", "yhat"]].copy()
            pdf_result[product_column] = pid
            pdf_result.rename(columns={"ds": date_column, "yhat": "forecast"}, inplace=True)
            results.append(pl.from_pandas(pdf_result))

        if not results:
            return pl.DataFrame()
        return pl.concat(results)

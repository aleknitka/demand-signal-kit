from abc import ABC, abstractmethod
import polars as pl


class BaseForecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(
        self,
        train_df: pl.DataFrame,
        date_column: str,
        target_column: str,
        product_column: str,
        feature_columns: list[str],
    ) -> None: ...

    @abstractmethod
    def predict(
        self,
        future_df: pl.DataFrame,
        date_column: str,
        product_column: str,
        feature_columns: list[str],
    ) -> pl.DataFrame: ...

    def get_params(self) -> dict:
        return {}

import polars as pl
from dataclasses import dataclass


REQUIRED_COLUMNS = ["date", "product_id", "quantity"]


@dataclass
class DataSchema:
    date_column: str = "date"
    product_column: str = "product_id"
    target_column: str = "quantity"
    event_columns: list[str] | None = None

    def validate(self, df: pl.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if not df[self.date_column].dtype == pl.Date:
            raise TypeError(f"Column '{self.date_column}' must be Date type")

    @property
    def required_columns(self) -> list[str]:
        return [self.date_column, self.product_column, self.target_column]


def validate_dataframe(df: pl.DataFrame, schema: DataSchema) -> pl.DataFrame:
    schema.validate(df)
    return df.sort([schema.product_column, schema.date_column])

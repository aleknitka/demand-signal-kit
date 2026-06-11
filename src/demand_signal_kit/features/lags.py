import polars as pl


def add_lag_features(
    df: pl.DataFrame,
    target_column: str = "quantity",
    product_column: str = "product_id",
    date_column: str = "date",
    lag_periods: list[int] | None = None,
) -> pl.DataFrame:
    if lag_periods is None:
        lag_periods = [7, 14, 28]

    lag_exprs = [
        pl.col(target_column)
        .shift(n)
        .over(product_column)
        .alias(f"lag_{n}")
        for n in lag_periods
    ]
    return df.with_columns(lag_exprs)

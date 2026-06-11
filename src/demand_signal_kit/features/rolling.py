import polars as pl


def add_rolling_features(
    df: pl.DataFrame,
    target_column: str = "quantity",
    product_column: str = "product_id",
    windows: list[int] | None = None,
    stats: list[str] | None = None,
) -> pl.DataFrame:
    if windows is None:
        windows = [7, 14, 28]
    if stats is None:
        stats = ["mean", "std"]

    exprs = []
    for w in windows:
        for stat in stats:
            col_name = f"rolling_{w}_{stat}"
            if stat == "mean":
                rolled = pl.col(target_column).rolling_mean(window_size=w)
            elif stat == "std":
                rolled = pl.col(target_column).rolling_std(window_size=w)
            elif stat == "min":
                rolled = pl.col(target_column).rolling_min(window_size=w)
            elif stat == "max":
                rolled = pl.col(target_column).rolling_max(window_size=w)
            else:
                continue
            exprs.append(rolled.over(product_column).alias(col_name))

    return df.with_columns(exprs)

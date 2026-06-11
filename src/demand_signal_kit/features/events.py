import polars as pl


def add_event_features(
    df: pl.DataFrame,
    event_columns: list[str],
) -> pl.DataFrame:
    for col in event_columns:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int8).fill_null(0))
    return df

import polars as pl


def add_cyclical_features(df: pl.DataFrame, date_column: str = "date") -> pl.DataFrame:
    import numpy as np

    return df.with_columns([
        (np.sin(2 * np.pi * pl.col(date_column).dt.weekday() / 7)).alias("dow_sin"),
        (np.cos(2 * np.pi * pl.col(date_column).dt.weekday() / 7)).alias("dow_cos"),
        (np.sin(2 * np.pi * pl.col(date_column).dt.month() / 12)).alias("month_sin"),
        (np.cos(2 * np.pi * pl.col(date_column).dt.month() / 12)).alias("month_cos"),
        (np.sin(2 * np.pi * pl.col(date_column).dt.quarter() / 4)).alias("quarter_sin"),
        (np.cos(2 * np.pi * pl.col(date_column).dt.quarter() / 4)).alias("quarter_cos"),
        (np.sin(2 * np.pi * pl.col(date_column).dt.day() / 30)).alias("day_sin"),
        (np.cos(2 * np.pi * pl.col(date_column).dt.day() / 30)).alias("day_cos"),
    ])

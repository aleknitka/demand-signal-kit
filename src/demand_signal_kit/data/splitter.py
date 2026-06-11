import polars as pl


def train_test_split(
    df: pl.DataFrame,
    date_column: str = "date",
    test_size: float = 0.2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    dates = df.select(date_column).unique().sort(date_column)
    split_idx = int(len(dates) * (1 - test_size))
    cutoff = dates[split_idx, date_column]

    train = df.filter(pl.col(date_column) < cutoff)
    test = df.filter(pl.col(date_column) >= cutoff)
    return train, test


def future_dates(
    last_date: str,
    horizon: int,
    product_ids: list[str],
) -> pl.DataFrame:
    from datetime import date, timedelta
    start = date.fromisoformat(last_date)
    dates = [start + timedelta(days=i + 1) for i in range(horizon)]
    rows = [{"date": d, "product_id": pid} for pid in product_ids for d in dates]
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

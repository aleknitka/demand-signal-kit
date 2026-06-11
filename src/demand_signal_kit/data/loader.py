from __future__ import annotations

import polars as pl
from sqlalchemy import create_engine

from demand_signal_kit.config import DataConfig
from demand_signal_kit.data.schema import DataSchema, validate_dataframe


def load_data(config: DataConfig) -> pl.DataFrame:
    if config.source_type == "csv":
        df = _load_csv(config.source_path)
    elif config.source_type == "parquet":
        df = _load_parquet(config.source_path)
    elif config.source_type == "postgres":
        df = _load_postgres(config.connection_string, config.source_path)
    else:
        raise ValueError(f"Unknown source_type: {config.source_type}")

    schema = DataSchema(
        date_column=config.date_column,
        product_column=config.product_column,
        target_column=config.target_column,
        event_columns=config.event_columns or [],
    )
    return validate_dataframe(df, schema)


def _load_csv(path: str) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True)


def _load_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def _load_postgres(conn_string: str, query_or_table: str) -> pl.DataFrame:
    engine = create_engine(conn_string)
    if query_or_table.strip().lower().startswith("select"):
        return pl.read_database(query=query_or_table, connection=engine)
    return pl.read_database(query=f"SELECT * FROM {query_or_table}", connection=engine)

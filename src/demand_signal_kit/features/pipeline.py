import polars as pl
from demand_signal_kit.config import FeatureConfig
from demand_signal_kit.features.cyclical import add_cyclical_features
from demand_signal_kit.features.holidays import add_holiday_features
from demand_signal_kit.features.events import add_event_features
from demand_signal_kit.features.lags import add_lag_features
from demand_signal_kit.features.rolling import add_rolling_features


def build_features(
    df: pl.DataFrame,
    config: FeatureConfig,
    date_column: str = "date",
    target_column: str = "quantity",
    product_column: str = "product_id",
) -> pl.DataFrame:
    df = df.with_columns(pl.col(date_column).cast(pl.Date))

    if config.include_cyclical:
        df = add_cyclical_features(df, date_column)

    if config.include_holidays:
        df = add_holiday_features(df, date_column, config.holiday_country)

    if config.custom_events:
        df = add_event_features(df, config.custom_events)

    df = add_lag_features(df, target_column, product_column, date_column, config.lag_periods)
    df = add_rolling_features(df, target_column, product_column, config.rolling_windows, config.rolling_stats)

    if "holiday_name" in df.columns:
        df = df.drop("holiday_name")

    return df.fill_null(0)

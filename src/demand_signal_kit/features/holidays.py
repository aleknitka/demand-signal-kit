import polars as pl
import holidays as holidays_lib


def add_holiday_features(
    df: pl.DataFrame,
    date_column: str = "date",
    country: str = "US",
) -> pl.DataFrame:
    hol = holidays_lib.country_holidays(country)
    dates = df[date_column].to_list()
    holiday_names = [hol.get(d, "") for d in dates]

    return df.with_columns([
        pl.Series("is_holiday_detected", [int(h != "") for h in holiday_names]),
        pl.Series("holiday_name", holiday_names),
    ])

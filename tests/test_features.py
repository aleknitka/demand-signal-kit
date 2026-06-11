import polars as pl
from datetime import date
from demand_signal_kit.features.pipeline import build_features
from demand_signal_kit.features.cyclical import add_cyclical_features
from demand_signal_kit.features.holidays import add_holiday_features
from demand_signal_kit.features.lags import add_lag_features
from demand_signal_kit.features.rolling import add_rolling_features
from demand_signal_kit.config import FeatureConfig
from demand_signal_kit.data.sample_generator import generate_sample_data


def test_cyclical_features():
    df = generate_sample_data(n_products=1, start_date=date(2024, 1, 1), end_date=date(2024, 1, 10))
    result = add_cyclical_features(df)
    assert "dow_sin" in result.columns
    assert "month_cos" in result.columns


def test_holiday_features():
    df = generate_sample_data(n_products=1, start_date=date(2024, 12, 24), end_date=date(2024, 12, 26))
    result = add_holiday_features(df)
    assert "is_holiday_detected" in result.columns
    assert "holiday_name" in result.columns


def test_lag_features():
    df = generate_sample_data(n_products=1, start_date=date(2024, 1, 1), end_date=date(2024, 2, 1))
    result = add_lag_features(df, lag_periods=[7])
    assert "lag_7" in result.columns


def test_rolling_features():
    df = generate_sample_data(n_products=1, start_date=date(2024, 1, 1), end_date=date(2024, 2, 1))
    result = add_rolling_features(df, windows=[7], stats=["mean"])
    assert "rolling_7_mean" in result.columns


def test_full_pipeline():
    df = generate_sample_data(n_products=2, start_date=date(2024, 1, 1), end_date=date(2024, 3, 1))
    config = FeatureConfig(lag_periods=[7], rolling_windows=[7], rolling_stats=["mean"])
    result = build_features(df, config)
    assert "dow_sin" in result.columns
    assert "is_holiday_detected" in result.columns
    assert "lag_7" in result.columns
    assert "rolling_7_mean" in result.columns

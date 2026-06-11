import polars as pl
from demand_signal_kit.evaluation.metrics import mae, rmse, mape, smape, evaluate_forecast


def test_mae():
    actual = pl.Series([10, 20, 30])
    predicted = pl.Series([12, 18, 33])
    assert abs(mae(actual, predicted) - 2.333) < 0.01


def test_rmse():
    actual = pl.Series([10, 20, 30])
    predicted = pl.Series([12, 18, 33])
    result = rmse(actual, predicted)
    assert result > 0


def test_mape():
    actual = pl.Series([100, 200, 300])
    predicted = pl.Series([110, 190, 330])
    result = mape(actual, predicted)
    assert result > 0


def test_evaluate_forecast():
    actual = pl.Series([10, 20, 30])
    predicted = pl.Series([10, 20, 30])
    result = evaluate_forecast(actual, predicted)
    assert result["MAE"] == 0
    assert result["RMSE"] == 0

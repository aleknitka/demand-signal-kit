import polars as pl
import numpy as np


def mae(actual: pl.Series, predicted: pl.Series) -> float:
    return float((actual - predicted).abs().mean())


def rmse(actual: pl.Series, predicted: pl.Series) -> float:
    return float(((actual - predicted) ** 2).mean()) ** 0.5


def mape(actual: pl.Series, predicted: pl.Series) -> float:
    df = pl.DataFrame({"actual": actual, "predicted": predicted}).filter(pl.col("actual") != 0)
    return float(((df["actual"] - df["predicted"]).abs() / df["actual"]).mean() * 100)


def smape(actual: pl.Series, predicted: pl.Series) -> float:
    df = pl.DataFrame({"actual": actual, "predicted": predicted})
    df = df.with_columns(((df["actual"].abs() + df["predicted"].abs()) / 2).alias("denom"))
    df = df.filter(pl.col("denom") != 0)
    return float(((df["actual"] - df["predicted"]).abs() / df["denom"]).mean() * 100)


def evaluate_forecast(
    actual: pl.Series,
    predicted: pl.Series,
) -> dict[str, float]:
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "SMAPE": smape(actual, predicted),
    }


def compare_models(
    results: dict[str, pl.DataFrame],
    date_column: str = "date",
    target_column: str = "quantity",
    forecast_column: str = "forecast",
    product_column: str = "product_id",
) -> pl.DataFrame:
    rows = []
    for model_name, df in results.items():
        metrics = evaluate_forecast(
            df[target_column],
            df[forecast_column],
        )
        metrics["model"] = model_name
        rows.append(metrics)

    return pl.DataFrame(rows).sort("MAE")

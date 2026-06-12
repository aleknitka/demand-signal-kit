import click
import polars as pl
from pathlib import Path
from demand_signal_kit.config import PipelineConfig, DataConfig, FeatureConfig, TrainingConfig
from demand_signal_kit.data.loader import load_data
from demand_signal_kit.data.sample_generator import generate_sample_data
from demand_signal_kit.data.splitter import train_test_split, future_dates
from demand_signal_kit.features.pipeline import build_features
from demand_signal_kit.models.registry import get_model, list_models
from demand_signal_kit.evaluation.metrics import compare_models

NUMERIC_TYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                 pl.Float32, pl.Float64}


def get_numeric_feature_cols(df: pl.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    return [c for c in df.columns if c not in exclude and df[c].dtype in NUMERIC_TYPES]


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Mimo Forecasting - Demand forecasting toolkit."""


@cli.command()
@click.option("--source", type=click.Choice(["csv", "parquet", "postgres"]), required=True)
@click.option("--path", "source_path", required=True)
@click.option("--date-col", default="date")
@click.option("--product-col", default="product_id")
@click.option("--target-col", default="quantity")
def ingest(source, source_path, date_col, product_col, target_col):
    """Load and validate data from a source."""
    config = DataConfig(
        source_type=source,
        source_path=source_path,
        date_column=date_col,
        product_column=product_col,
        target_column=target_col,
    )
    df = load_data(config)
    click.echo(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    click.echo(f"Products: {df[product_col].n_unique()}")
    click.echo(f"Date range: {df[date_col].min()} to {df[date_col].max()}")


@cli.command()
@click.option("--source", type=click.Choice(["csv", "parquet", "postgres"]), required=True)
@click.option("--path", "source_path", required=True)
@click.option("--model", "-m", multiple=True, default=["lightgbm"])
@click.option("--horizon", type=int, default=14)
@click.option("--test-size", type=float, default=0.2)
def train(source, source_path, model, horizon, test_size):
    """Train models and save evaluation."""
    config = DataConfig(source_type=source, source_path=source_path)
    df = load_data(config)
    feat_config = FeatureConfig()
    df = build_features(df, feat_config)
    train_df, test_df = train_test_split(df, test_size=test_size)

    feature_cols = get_numeric_feature_cols(df, exclude=("date", "product_id", "quantity"))
    results = {}

    for m in model:
        click.echo(f"Training {m}...")
        forecaster = get_model(m)()
        forecaster.fit(train_df, "date", "quantity", "product_id", feature_cols)
        preds = forecaster.predict(test_df, "date", "product_id", feature_cols)
        merged = test_df.select(["date", "product_id", "quantity"]).join(
            preds, on=["date", "product_id"]
        )
        results[m] = merged

    comparison = compare_models(results)
    click.echo("\nModel Comparison:")
    click.echo(comparison)


@cli.command()
@click.option("--source", type=click.Choice(["csv", "parquet", "postgres"]), required=True)
@click.option("--path", "source_path", required=True)
@click.option("--model", "-m", multiple=True, default=["lightgbm"])
@click.option("--horizon", type=int, default=14)
@click.option("--output", type=click.Path(), default="forecast_output.csv")
def predict(source, source_path, model, horizon, output):
    """Generate forecasts for future dates."""
    config = DataConfig(source_type=source, source_path=source_path)
    df = load_data(config)
    feat_config = FeatureConfig()
    df = build_features(df, feat_config)

    last_date = df["date"].max().isoformat()
    products = df["product_id"].unique().to_list()
    future = future_dates(last_date, horizon, products)
    future = build_features(pl.concat([df, future]), feat_config).filter(
        pl.col("date") > last_date
    )

    feature_cols = get_numeric_feature_cols(future, exclude=("date", "product_id", "quantity"))

    for m in model:
        click.echo(f"Predicting with {m}...")
        forecaster = get_model(m)()
        forecaster.fit(df, "date", "quantity", "product_id", feature_cols)
        preds = forecaster.predict(future, "date", "product_id", feature_cols)
        preds.write_csv(output)
        click.echo(f"Saved to {output}")


@cli.command()
def demo():
    """Run full pipeline with sample data."""
    click.echo("Generating sample data...")
    df = generate_sample_data()
    click.echo(f"Generated {len(df)} rows")

    feat_config = FeatureConfig()
    df = build_features(df, feat_config)
    train_df, test_df = train_test_split(df)

    feature_cols = get_numeric_feature_cols(df, exclude=("date", "product_id", "quantity"))

    click.echo("\nTraining LightGBM...")
    model = get_model("lightgbm")()
    model.fit(train_df, "date", "quantity", "product_id", feature_cols)
    preds = model.predict(test_df, "date", "product_id", feature_cols)
    merged = test_df.select(["date", "product_id", "quantity"]).join(
        preds, on=["date", "product_id"]
    )

    from demand_signal_kit.evaluation.metrics import evaluate_forecast
    metrics = evaluate_forecast(merged["quantity"], merged["forecast"])
    click.echo("\nEvaluation Metrics:")
    for k, v in metrics.items():
        click.echo(f"  {k}: {v:.2f}")

    click.echo("\nPipeline complete!")


@cli.command()
def models():
    """List available models."""
    click.echo("Available models:")
    for m in list_models():
        click.echo(f"  - {m}")


@cli.command()
@click.option("--stores", type=int, default=10, help="Number of stores")
@click.option("--products", type=int, default=100, help="Number of products")
@click.option("--start-date", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--format", "output_format", type=click.Choice(["parquet", "csv"]), default="parquet")
@click.option("--output", type=click.Path(), default="data/synthetic", help="Output directory")
@click.option("--flat", is_flag=True, help="Also generate flat forecast-ready file")
def generate(stores, products, start_date, end_date, seed, output_format, output, flat):
    """Generate synthetic sales data with stores, products, promos, and receipts."""
    from datetime import date as date_type
    from demand_signal_kit.data.synthetic import SyntheticDataGenerator, SyntheticDataConfig

    config = SyntheticDataConfig(
        n_stores=stores,
        n_products=products,
        date_start=date_type.fromisoformat(start_date),
        date_end=date_type.fromisoformat(end_date),
        seed=seed,
        output_format=output_format,
    )

    click.echo(f"Generating synthetic data: {stores} stores, {products} products...")
    gen = SyntheticDataGenerator(config)
    gen.generate()

    if output_format == "parquet":
        out = gen.write_parquet(output)
    else:
        out = gen.write_csv(output)

    summary = gen.summary()
    click.echo(f"\nGenerated in {out}/")
    click.echo(f"  Stores:          {summary['stores']}")
    click.echo(f"  Products:        {summary['products']}")
    click.echo(f"  Promos:          {summary['promos']}")
    click.echo(f"  Receipts:        {summary['receipts']:,}")
    click.echo(f"  Receipt items:   {summary['receipt_items']:,}")
    click.echo(f"  Date range:      {summary['date_range']}")
    click.echo(f"  Total revenue:   ${summary['total_revenue']:,.2f}")
    click.echo(f"  Avg basket size: {summary['avg_basket_size']} items")

    if flat:
        forecast = gen.to_forecast_frame()
        flat_path = out / f"forecast_input.{output_format}"
        if output_format == "parquet":
            forecast.write_parquet(flat_path)
        else:
            forecast.write_csv(flat_path)
        click.echo(f"\nFlat forecast file: {flat_path} ({len(forecast):,} rows)")

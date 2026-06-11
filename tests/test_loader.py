import polars as pl
from datetime import date
from demand_signal_kit.data.loader import load_data
from demand_signal_kit.config import DataConfig
from demand_signal_kit.data.sample_generator import generate_sample_data


def test_generate_sample_data():
    df = generate_sample_data(n_products=2, start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
    assert len(df) > 0
    assert "date" in df.columns
    assert "product_id" in df.columns
    assert "quantity" in df.columns
    assert df["product_id"].n_unique() == 2


def test_load_csv(tmp_path):
    df = generate_sample_data(n_products=1, start_date=date(2024, 1, 1), end_date=date(2024, 1, 10))
    csv_path = tmp_path / "test.csv"
    df.write_csv(csv_path)

    config = DataConfig(source_type="csv", source_path=str(csv_path))
    loaded = load_data(config)
    assert len(loaded) == 10


def test_load_parquet(tmp_path):
    df = generate_sample_data(n_products=1, start_date=date(2024, 1, 1), end_date=date(2024, 1, 10))
    pq_path = tmp_path / "test.parquet"
    df.write_parquet(pq_path)

    config = DataConfig(source_type="parquet", source_path=str(pq_path))
    loaded = load_data(config)
    assert len(loaded) == 10

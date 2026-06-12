import polars as pl
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from demand_signal_kit.data.synthetic.config import SyntheticDataConfig
from demand_signal_kit.data.synthetic.stores import generate_stores
from demand_signal_kit.data.synthetic.products import generate_products
from demand_signal_kit.data.synthetic.promos import generate_promos
from demand_signal_kit.data.synthetic.receipts import generate_receipts
from demand_signal_kit.data.synthetic.compatibility import (
    to_forecast_frame,
    flatten_to_daily_demand,
    flatten_to_product_daily,
)


@dataclass
class SyntheticData:
    stores: pl.DataFrame
    products: pl.DataFrame
    promos: pl.DataFrame
    receipts: pl.DataFrame
    receipt_items: pl.DataFrame


class SyntheticDataGenerator:
    def __init__(self, config: SyntheticDataConfig | None = None):
        self.config = config or SyntheticDataConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._data: SyntheticData | None = None

    def generate(self) -> SyntheticData:
        stores = generate_stores(self.config, self.rng)
        products = generate_products(self.config, self.rng)
        promos = generate_promos(self.config, products, stores, self.rng)
        receipts, receipt_items, promo_spend = generate_receipts(
            self.config, stores, products, promos, self.rng
        )

        promos = promos.join(
            promo_spend, on="promo_id", how="left"
        ).with_columns(
            pl.col("actual_spend").fill_null(0)
        )

        self._data = SyntheticData(
            stores=stores,
            products=products,
            promos=promos,
            receipts=receipts,
            receipt_items=receipt_items,
        )
        return self._data

    @property
    def data(self) -> SyntheticData:
        if self._data is None:
            raise RuntimeError("Call generate() first")
        return self._data

    def to_forecast_frame(self) -> pl.DataFrame:
        return to_forecast_frame(
            self.data.receipts,
            self.data.receipt_items,
            self.data.products,
            self.data.promos,
        )

    def to_daily_store_product(self) -> pl.DataFrame:
        return flatten_to_daily_demand(self.data.receipts, self.data.receipt_items)

    def to_daily_product(self) -> pl.DataFrame:
        return flatten_to_product_daily(self.data.receipts, self.data.receipt_items)

    def write_parquet(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.data.stores.write_parquet(output_dir / "stores.parquet")
        self.data.products.write_parquet(output_dir / "products.parquet")
        self.data.promos.write_parquet(output_dir / "promos.parquet")
        self.data.receipts.write_parquet(output_dir / "receipts.parquet")
        self.data.receipt_items.write_parquet(output_dir / "receipt_items.parquet")

        forecast = self.to_forecast_frame()
        forecast.write_parquet(output_dir / "forecast_input.parquet")

        return output_dir

    def write_csv(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.data.stores.write_csv(output_dir / "stores.csv")
        self.data.products.write_csv(output_dir / "products.csv")
        self.data.promos.write_csv(output_dir / "promos.csv")
        self.data.receipts.write_csv(output_dir / "receipts.csv")
        self.data.receipt_items.write_csv(output_dir / "receipt_items.csv")

        forecast = self.to_forecast_frame()
        forecast.write_csv(output_dir / "forecast_input.csv")

        return output_dir

    def summary(self) -> dict:
        d = self.data
        return {
            "stores": len(d.stores),
            "products": len(d.products),
            "promos": len(d.promos),
            "receipts": len(d.receipts),
            "receipt_items": len(d.receipt_items),
            "date_range": f"{self.config.date_start} → {self.config.date_end}",
            "total_revenue": round(d.receipts["total"].sum(), 2),
            "avg_basket_size": round(d.receipts["items_count"].mean(), 1),
        }

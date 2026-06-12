from dataclasses import dataclass, field
from demand_signal_kit.data.synthetic.generator import SyntheticDataGenerator, SyntheticData
from demand_signal_kit.data.synthetic.config import SyntheticDataConfig


@dataclass
class Scenario:
    name: str
    description: str
    config_overrides: dict = field(default_factory=dict)


def run_scenario(scenario: Scenario, base_config: SyntheticDataConfig | None = None) -> SyntheticData:
    config = base_config or SyntheticDataConfig()
    for key, value in scenario.config_overrides.items():
        setattr(config, key, value)

    gen = SyntheticDataGenerator(config)
    return gen.generate()


def compare_scenarios(
    scenarios: list[Scenario],
    base_config: SyntheticDataConfig | None = None,
) -> list[dict]:
    results = []
    baseline_data = None

    for i, scenario in enumerate(scenarios):
        data = run_scenario(scenario, base_config)

        daily_product = (
            data.receipt_items
            .join(data.receipts.select("receipt_id", "transaction_date"), on="receipt_id")
            .filter(~pl.col("is_returned"))
            .group_by("transaction_date", "product_id")
            .agg(pl.col("quantity").sum().alias("quantity"))
        )

        total_qty = daily_product["quantity"].sum()
        total_revenue = data.receipts["total"].sum()
        n_products = data.products["product_id"].n_unique()
        avg_daily_receipts = len(data.receipts) / max(1, (
            base_config or SyntheticDataConfig()
        ).n_stores)

        result = {
            "scenario": scenario.name,
            "description": scenario.description,
            "total_quantity": int(total_qty),
            "total_revenue": round(float(total_revenue), 2),
            "unique_products_sold": n_products,
            "avg_daily_receipts_per_store": round(float(avg_daily_receipts), 1),
            "n_receipts": len(data.receipts),
            "n_stores": len(data.stores),
        }

        if i == 0:
            baseline_data = result
        else:
            for key in ["total_quantity", "total_revenue", "n_receipts"]:
                result[f"{key}_delta"] = round(result[key] - baseline_data[key], 2)
                result[f"{key}_pct"] = round(
                    (result[key] - baseline_data[key]) / max(1, baseline_data[key]) * 100, 1
                )

        results.append(result)

    return results

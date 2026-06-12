import polars as pl
import numpy as np
from datetime import date, timedelta

from demand_signal_kit.data.synthetic.config import SyntheticDataConfig, PROMO_TYPES


def generate_promos(
    config: SyntheticDataConfig,
    products: pl.DataFrame,
    stores: pl.DataFrame,
    rng: np.random.Generator,
) -> pl.DataFrame:
    n_days = (config.date_end - config.date_start).days + 1
    total_product_days = config.n_products * n_days
    n_promos = int(total_product_days * config.promo_rate / config.avg_promo_duration)

    promo_type_names = list(PROMO_TYPES.keys())
    promo_types = rng.choice(promo_type_names, size=n_promos)

    categories = products["category"].unique().to_list()
    store_types = stores["store_type"].unique().to_list()
    store_ids = stores["store_id"].to_list()

    scope_choices = ["product", "category", "store", "store_category"]
    scope_weights = [0.4, 0.3, 0.15, 0.15]
    scopes = rng.choice(scope_choices, size=n_promos, p=scope_weights)

    rows = []
    for i in range(n_promos):
        pt = promo_types[i]
        scope = scopes[i]
        promo_info = PROMO_TYPES[pt]

        discount = round(float(rng.uniform(*promo_info["discount_range"])), 3)
        duration = int(rng.integers(3, 22))
        start_day = int(rng.integers(0, n_days - duration))
        start = config.date_start + timedelta(days=start_day)
        end = start + timedelta(days=duration - 1)

        if scope == "product":
            target = products["product_id"][int(rng.integers(0, config.n_products))]
        elif scope == "category":
            target = rng.choice(categories)
        elif scope == "store":
            target = rng.choice(store_ids)
        else:
            target = rng.choice(store_types)

        rows.append({
            "promo_id": f"PR{i+1:06d}",
            "promo_name": f"{pt.replace('_', ' ').title()} #{i+1}",
            "promo_type": pt,
            "discount_value": discount,
            "start_date": start,
            "end_date": end,
            "duration_days": duration,
            "scope_type": scope,
            "scope_target": target,
            "budget_cap": round(float(rng.uniform(5000, 100000)), 2),
            "actual_spend": 0.0,
            "is_national": bool(rng.random() < 0.3),
            "priority": int(rng.integers(1, 6)),
        })

    return pl.DataFrame(rows).with_columns(
        pl.col("start_date").cast(pl.Date),
        pl.col("end_date").cast(pl.Date),
    )

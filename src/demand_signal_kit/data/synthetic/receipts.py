import polars as pl
import numpy as np
from datetime import date, timedelta, time

from demand_signal_kit.data.synthetic.config import SyntheticDataConfig, CATEGORY_PROFILES, US_HOLIDAYS_2022_2024


def compute_demand_modifier(
    product: dict,
    store: dict,
    day: date,
    active_promos: list[dict],
    rng: np.random.Generator,
) -> float:
    modifier = 1.0

    doy = day.timetuple().tm_yday
    if product["is_seasonal"]:
        seasonal = product["seasonal_amplitude"] * np.sin(
            2 * np.pi * (doy - product["seasonal_peak_day"]) / 365
        )
        modifier *= (1 + seasonal)

    dow = day.weekday()
    cat = product["category"]
    weekly = CATEGORY_PROFILES[cat]["weekly_pattern"]
    modifier *= weekly[dow]

    is_holiday = day in US_HOLIDAYS_2022_2024
    is_black_friday = day.month == 11 and 22 <= day.day <= 30 and day.weekday() == 4
    if is_holiday or is_black_friday:
        lift = CATEGORY_PROFILES[cat]["holiday_lift"]
        modifier *= (1 + lift * 0.3)

    for promo in active_promos:
        if promo["scope_type"] == "product" and promo["scope_target"] == product["product_id"]:
            promo_type = promo["promo_type"]
            from demand_signal_kit.data.synthetic.config import PROMO_TYPES
            lift_range = PROMO_TYPES[promo_type]["lift_range"]
            lift = float(rng.uniform(*lift_range))
            discount_effect = promo["discount_value"] * product["price_elasticity"]
            modifier *= (1 + lift * 0.3 - discount_effect * 0.1)
        elif promo["scope_type"] == "category" and promo["scope_target"] == product["category"]:
            modifier *= 1.2
        elif promo["scope_type"] == "store" and promo["scope_target"] == store["store_id"]:
            modifier *= 1.1
        elif promo["scope_type"] == "store_category" and promo["scope_target"] == store["store_type"]:
            modifier *= 1.15

    modifier *= store["promo_response_rate"]

    noise = float(rng.normal(1.0, 0.1))
    modifier *= max(0.1, noise)

    return modifier


def generate_receipts(
    config: SyntheticDataConfig,
    stores: pl.DataFrame,
    products: pl.DataFrame,
    promos: pl.DataFrame,
    rng: np.random.Generator,
    customers: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    stores_dict = stores.to_dicts()
    products_dict = products.to_dicts()
    promos_dict = promos.to_dicts()
    customers_dict = customers.to_dicts() if customers is not None else []
    has_customers = len(customers_dict) > 0

    n_days = (config.date_end - config.date_start).days + 1
    min_receipts, max_receipts = config.receipts_per_store_per_day

    receipt_rows = []
    item_rows = []
    receipt_counter = 0

    product_weights = np.array([p["base_demand"] for p in products_dict])
    product_weights = product_weights / product_weights.sum()

    for store in stores_dict:
        store_traffic = store["weekly_foot_traffic"]
        base_daily = store_traffic / 7

        for day_offset in range(n_days):
            day = config.date_start + timedelta(days=day_offset)
            dow = day.weekday()

            traffic_mod = 1.3 if dow >= 5 else 1.0
            is_holiday = day in US_HOLIDAYS_2022_2024
            if is_holiday:
                traffic_mod *= 0.7

            daily_receipts = int(rng.uniform(
                min_receipts * traffic_mod * 0.5,
                max_receipts * traffic_mod * max(0.5, base_daily / 5000),
            ))
            daily_receipts = max(5, min(daily_receipts, 500))

            active_promos = [
                p for p in promos_dict
                if p["start_date"] <= day <= p["end_date"]
                and (p["is_national"]
                     or p["scope_type"] == "store" and p["scope_target"] == store["store_id"]
                     or p["scope_type"] == "store_category" and p["scope_target"] == store["store_type"])
            ]

            for _ in range(daily_receipts):
                receipt_counter += 1
                receipt_id = f"R{receipt_counter:08d}"

                customer_id = ""
                customer_seg = ""
                loyalty_tier = ""
                if has_customers:
                    cust = customers_dict[int(rng.integers(0, len(customers_dict)))]
                    customer_id = cust["customer_id"]
                    customer_seg = cust["segment"]
                    loyalty_tier = cust["loyalty_tier"]

                hour = int(rng.choice(
                    range(24),
                    p=_time_distribution(dow),
                ))
                minute = int(rng.integers(0, 60))
                trans_time = time(hour, minute)

                n_items = int(rng.integers(1, 13))
                if has_customers:
                    from demand_signal_kit.data.synthetic.customers import get_segment_basket_size
                    n_items = get_segment_basket_size(cust, rng)

                selected_indices = rng.choice(
                    len(products_dict),
                    size=n_items,
                    p=product_weights,
                    replace=True,
                )

                payment = rng.choice(
                    ["cash", "credit", "debit", "mobile"],
                    p=[0.15, 0.40, 0.30, 0.15],
                )

                subtotal = 0.0
                total_discount = 0.0

                loyalty_discount = 0.0
                if has_customers and loyalty_tier != "none":
                    from demand_signal_kit.data.synthetic.customers import LOYALTY_TIERS
                    loyalty_discount = LOYALTY_TIERS[loyalty_tier]["discount_pct"]

                for idx in selected_indices:
                    product = products_dict[idx]
                    modifier = compute_demand_modifier(product, store, day, active_promos, rng)
                    qty = max(1, int(rng.poisson(1.5 * modifier)))

                    price = product["unit_price"]
                    discount = 0.0
                    applied_promo_id = None

                    for promo in active_promos:
                        if (promo["scope_type"] == "product" and promo["scope_target"] == product["product_id"]) or \
                           (promo["scope_type"] == "category" and promo["scope_target"] == product["category"]):
                            if promo["promo_type"] == "percentage_off":
                                discount = price * promo["discount_value"]
                            elif promo["promo_type"] == "fixed_discount":
                                discount = min(promo["discount_value"], price * 0.5)
                            elif promo["promo_type"] == "clearance":
                                discount = price * promo["discount_value"]
                            applied_promo_id = promo["promo_id"]
                            break

                    actual_price = max(0.01, price - discount)
                    if loyalty_discount > 0:
                        actual_price = max(0.01, actual_price * (1 - loyalty_discount))
                    line_total = round(qty * actual_price, 2)
                    subtotal += line_total
                    total_discount += discount * qty

                    item_rows.append({
                        "receipt_id": receipt_id,
                        "product_id": product["product_id"],
                        "quantity": qty,
                        "unit_price": round(actual_price, 2),
                        "line_total": line_total,
                        "discount_applied": round(discount * qty, 2),
                        "promo_id": applied_promo_id or "",
                        "is_returned": bool(rng.random() < CATEGORY_PROFILES[product["category"]]["return_rate"]),
                    })

                tax_rate = 0.08
                tax = round(subtotal * tax_rate, 2)
                total = round(subtotal + tax, 2)

                basket = "small" if n_items <= 3 else ("medium" if n_items <= 7 else "large")

                receipt_rows.append({
                    "receipt_id": receipt_id,
                    "store_id": store["store_id"],
                    "customer_id": customer_id,
                    "transaction_date": day,
                    "transaction_time": trans_time,
                    "subtotal": round(subtotal, 2),
                    "tax": tax,
                    "total": total,
                    "payment_method": payment,
                    "items_count": n_items,
                    "is_weekend": dow >= 5,
                    "day_of_week": dow,
                    "hour_of_day": hour,
                    "basket_size": basket,
                    "customer_segment": customer_seg,
                    "loyalty_tier": loyalty_tier,
                })

    receipts_df = pl.DataFrame(receipt_rows).with_columns(
        pl.col("transaction_date").cast(pl.Date),
    )
    items_df = pl.DataFrame(item_rows)

    promo_spend = items_df.group_by("promo_id").agg(
        pl.col("discount_applied").sum().alias("actual_spend")
    ).filter(pl.col("promo_id").is_not_null())

    return receipts_df, items_df, promo_spend


def _time_distribution(dow: int) -> list[float]:
    weights = [
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.06,
        0.07, 0.06, 0.07, 0.09, 0.10, 0.08, 0.06, 0.05,
        0.05, 0.06, 0.07, 0.06, 0.04, 0.03, 0.02, 0.01,
    ]
    if dow >= 5:
        weights[9] += 0.03
        weights[10] += 0.04
        weights[14] += 0.03
        weights[15] += 0.02
    total = sum(weights)
    return [w / total for w in weights]

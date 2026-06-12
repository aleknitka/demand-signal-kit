import polars as pl


def flatten_to_daily_demand(
    receipts: pl.DataFrame,
    receipt_items: pl.DataFrame,
) -> pl.DataFrame:
    items_with_date = receipt_items.join(
        receipts.select("receipt_id", "store_id", "transaction_date"),
        on="receipt_id",
    )

    daily = (
        items_with_date
        .filter(~pl.col("is_returned"))
        .group_by("transaction_date", "store_id", "product_id")
        .agg([
            pl.col("quantity").sum().alias("quantity"),
            pl.col("line_total").sum().alias("revenue"),
            pl.col("discount_applied").sum().alias("total_discount"),
            pl.col("receipt_id").n_unique().alias("transaction_count"),
        ])
    )

    return daily.rename({"transaction_date": "date"})


def flatten_to_product_daily(
    receipts: pl.DataFrame,
    receipt_items: pl.DataFrame,
) -> pl.DataFrame:
    items_with_date = receipt_items.join(
        receipts.select("receipt_id", "store_id", "transaction_date"),
        on="receipt_id",
    )

    daily = (
        items_with_date
        .filter(~pl.col("is_returned"))
        .group_by("transaction_date", "product_id")
        .agg([
            pl.col("quantity").sum().alias("quantity"),
            pl.col("line_total").sum().alias("revenue"),
            pl.col("discount_applied").sum().alias("total_discount"),
            pl.col("store_id").n_unique().alias("store_count"),
            pl.col("receipt_id").n_unique().alias("transaction_count"),
        ])
    )

    return daily.rename({"transaction_date": "date"})


def to_forecast_frame(
    receipts: pl.DataFrame,
    receipt_items: pl.DataFrame,
    products: pl.DataFrame,
    promos: pl.DataFrame,
) -> pl.DataFrame:
    daily_store_product = flatten_to_daily_demand(receipts, receipt_items)

    promo_flags = _build_promo_flags(promos, daily_store_product)

    result = daily_store_product.join(
        products.select("product_id", "category", "unit_price", "margin"),
        on="product_id",
        how="left",
    )

    if promo_flags is not None and len(promo_flags) > 0:
        result = result.join(promo_flags, on=["date", "product_id"], how="left")

    result = result.with_columns(
        pl.col("total_discount").fill_null(0),
        pl.col("has_promo").fill_null(False).cast(pl.Boolean),
    )

    return result.sort(["date", "store_id", "product_id"])


def _build_promo_flags(promos: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame | None:
    if len(promos) == 0:
        return None

    product_promos = promos.filter(pl.col("scope_type") == "product")
    if len(product_promos) == 0:
        return None

    rows = []
    for promo in product_promos.to_dicts():
        d = promo["start_date"]
        while d <= promo["end_date"]:
            rows.append({"date": d, "product_id": promo["scope_target"], "has_promo": True})
            from datetime import timedelta
            d += timedelta(days=1)

    if not rows:
        return None

    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

import polars as pl
import numpy as np

from demand_signal_kit.data.synthetic.config import SyntheticDataConfig, CATEGORY_PROFILES


SUBCATEGORIES = {
    "Electronics": ["Phones", "Laptops", "Tablets", "Headphones", "Cameras", "Speakers", "Monitors", "Keyboards"],
    "Clothing": ["T-Shirts", "Jeans", "Jackets", "Dresses", "Sneakers", "Boots", "Shirts", "Shorts"],
    "Food & Beverage": ["Snacks", "Beverages", "Dairy", "Bakery", "Frozen", "Canned", "Fresh", "Condiments"],
    "Home & Garden": ["Furniture", "Lighting", "Kitchen", "Bedding", "Tools", "Plants", "Decor", "Storage"],
    "Sports & Outdoors": ["Running", "Cycling", "Camping", "Swimming", "Yoga", "Team Sports", "Hiking", "Fishing"],
}

BRANDS = {
    "Electronics": ["TechVault", "PixelPro", "SoundMax", "ByteGear", "VoltEdge", "ScreenStar"],
    "Clothing": ["UrbanThread", "DenimWorks", "StyleCraft", "FabricFusion", "WearWell", "ThreadLine"],
    "Food & Beverage": ["NatureBite", "FreshCrunch", "GoldenGrain", "PureSip", "HarvestPack", "TastyLab"],
    "Home & Garden": ["HomeCraft", "GardenEase", "BrightSpace", "CozyNest", "ToolForge", "GreenThumb"],
    "Sports & Outdoors": ["TrailBlaze", "PeakFit", "WaveRider", "SummitGear", "ActivePulse", "OutdoorX"],
}

LIFECYCLE_STAGES = ["launch", "growth", "mature", "decline"]


def generate_products(config: SyntheticDataConfig, rng: np.random.Generator) -> pl.DataFrame:
    n = config.n_products

    cat_weights = list(config.category_weights.values())
    cat_names = list(config.category_weights.keys())
    categories = rng.choice(cat_names, size=n, p=cat_weights)

    rows = []
    for i in range(n):
        cat = categories[i]
        profile = CATEGORY_PROFILES[cat]

        subcat = rng.choice(SUBCATEGORIES[cat])
        brand = rng.choice(BRANDS[cat])

        price = round(float(rng.uniform(*profile["price_range"])), 2)
        margin = round(float(rng.uniform(*profile["margin_range"])), 3)
        cost = round(price * (1 - margin), 2)

        base_demand = round(float(rng.uniform(*profile["base_demand_range"])), 2)
        elasticity = round(float(rng.uniform(*profile["elasticity_range"])), 3)
        promo_sens = round(float(rng.uniform(*profile["promo_sensitivity"])), 3)

        is_seasonal = bool(rng.random() < 0.6)
        seasonal_peak = int(rng.integers(1, 366))
        seasonal_amp = round(float(rng.uniform(0.2, 0.9)), 3) if is_seasonal else 0.0

        lifecycle = rng.choice(LIFECYCLE_STAGES, p=[0.1, 0.2, 0.5, 0.2])

        cannibal_group = f"{cat}_{subcat}_{i // 5}"

        product_id = f"P{i+1:05d}"
        product_name = f"{brand} {subcat} {i+1:04d}"

        rows.append({
            "product_id": product_id,
            "product_name": product_name,
            "category": cat,
            "subcategory": subcat,
            "brand": brand,
            "unit_price": price,
            "cost": cost,
            "margin": margin,
            "weight_kg": round(float(rng.uniform(0.1, 20)), 2),
            "is_seasonal": is_seasonal,
            "seasonal_peak_day": seasonal_peak,
            "seasonal_amplitude": seasonal_amp,
            "base_demand": base_demand,
            "price_elasticity": elasticity,
            "promo_sensitivity": promo_sens,
            "cannibalization_group": cannibal_group,
            "lifecycle_stage": lifecycle,
        })

    return pl.DataFrame(rows)

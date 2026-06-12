from dataclasses import dataclass, field
from datetime import date


@dataclass
class SyntheticDataConfig:
    n_stores: int = 200
    n_products: int = 5000
    date_start: date = date(2022, 1, 1)
    date_end: date = date(2024, 12, 31)
    seed: int = 42

    store_type_weights: dict[str, float] = field(default_factory=lambda: {
        "Grocery": 0.35,
        "Department": 0.15,
        "Convenience": 0.25,
        "Pharmacy": 0.15,
        "Specialty": 0.10,
    })

    receipts_per_store_per_day: tuple[float, float] = (30, 200)
    promo_rate: float = 0.08
    avg_promo_duration: int = 10
    output_format: str = "parquet"

    category_weights: dict[str, float] = field(default_factory=lambda: {
        "Electronics": 0.15,
        "Clothing": 0.20,
        "Food & Beverage": 0.30,
        "Home & Garden": 0.15,
        "Sports & Outdoors": 0.20,
    })

    n_customers: int = 50000
    loyalty_enrollment_rate: float = 0.6
    customer_segment_weights: dict[str, float] = field(default_factory=lambda: {
        "budget": 0.25,
        "regular": 0.35,
        "premium": 0.20,
        "vip": 0.10,
        "inactive": 0.10,
    })


STORE_TYPE_SIZE_PROFILES = {
    ("Grocery", "Express"): {"sqft": (5000, 12000), "traffic": (800, 1500)},
    ("Grocery", "Standard"): {"sqft": (15000, 35000), "traffic": (2000, 5000)},
    ("Grocery", "Large"): {"sqft": (40000, 70000), "traffic": (5000, 10000)},
    ("Department", "Standard"): {"sqft": (20000, 50000), "traffic": (1000, 3000)},
    ("Department", "Flagship"): {"sqft": (80000, 150000), "traffic": (3000, 8000)},
    ("Convenience", "Express"): {"sqft": (2000, 5000), "traffic": (500, 1200)},
    ("Convenience", "Standard"): {"sqft": (3000, 8000), "traffic": (600, 1500)},
    ("Pharmacy", "Standard"): {"sqft": (10000, 20000), "traffic": (800, 2000)},
    ("Pharmacy", "Express"): {"sqft": (3000, 7000), "traffic": (400, 900)},
    ("Specialty", "Standard"): {"sqft": (5000, 15000), "traffic": (300, 800)},
    ("Specialty", "Flagship"): {"sqft": (15000, 30000), "traffic": (600, 1500)},
}

STORE_TYPE_SIZES = {
    "Grocery": ["Express", "Standard", "Large"],
    "Department": ["Standard", "Flagship"],
    "Convenience": ["Express", "Standard"],
    "Pharmacy": ["Standard", "Express"],
    "Specialty": ["Standard", "Flagship"],
}

REGIONS = {
    "Northeast": {"states": ["NY", "NJ", "PA", "CT", "MA", "ME", "VT", "NH", "RI"], "traffic_mod": 1.1},
    "Southeast": {"states": ["FL", "GA", "NC", "SC", "VA", "TN", "AL", "MS"], "traffic_mod": 1.0},
    "Midwest": {"states": ["OH", "MI", "IL", "IN", "WI", "MN", "IA", "MO"], "traffic_mod": 0.95},
    "Southwest": {"states": ["TX", "AZ", "NM", "OK", "NV"], "traffic_mod": 1.0},
    "West": {"states": ["CA", "OR", "WA", "CO", "UT"], "traffic_mod": 1.05},
}

REGION_CITIES = {
    "Northeast": ["New York", "Boston", "Philadelphia", "Hartford", "Providence", "Portland"],
    "Southeast": ["Miami", "Atlanta", "Charlotte", "Nashville", "Tampa", "Raleigh"],
    "Midwest": ["Chicago", "Detroit", "Minneapolis", "Columbus", "Indianapolis", "Milwaukee"],
    "Southwest": ["Houston", "Dallas", "Phoenix", "San Antonio", "Austin", "Oklahoma City"],
    "West": ["Los Angeles", "San Francisco", "Seattle", "Denver", "Portland", "Las Vegas"],
}

CATEGORY_PROFILES = {
    "Electronics": {
        "price_range": (15, 2000),
        "margin_range": (0.10, 0.35),
        "base_demand_range": (0.5, 8),
        "elasticity_range": (-2.5, -1.5),
        "promo_sensitivity": (0.3, 0.7),
        "weekly_pattern": [0.8, 0.7, 0.9, 1.0, 1.2, 1.5, 1.1],
        "yearly_amplitude": 25,
        "yearly_peak_day": 340,
        "promo_lift": 1.8,
        "holiday_lift": 2.5,
        "return_rate": 0.05,
    },
    "Clothing": {
        "price_range": (10, 300),
        "margin_range": (0.25, 0.60),
        "base_demand_range": (1, 15),
        "elasticity_range": (-2.0, -1.0),
        "promo_sensitivity": (0.4, 0.8),
        "weekly_pattern": [0.6, 0.7, 0.8, 0.9, 1.1, 1.6, 1.3],
        "yearly_amplitude": 30,
        "yearly_peak_day": 330,
        "promo_lift": 2.0,
        "holiday_lift": 1.5,
        "return_rate": 0.08,
    },
    "Food & Beverage": {
        "price_range": (1, 50),
        "margin_range": (0.05, 0.25),
        "base_demand_range": (5, 40),
        "elasticity_range": (-1.5, -0.5),
        "promo_sensitivity": (0.2, 0.5),
        "weekly_pattern": [0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 0.8],
        "yearly_amplitude": 15,
        "yearly_peak_day": 355,
        "promo_lift": 1.4,
        "holiday_lift": 1.8,
        "return_rate": 0.01,
    },
    "Home & Garden": {
        "price_range": (5, 500),
        "margin_range": (0.15, 0.45),
        "base_demand_range": (0.3, 6),
        "elasticity_range": (-1.8, -0.8),
        "promo_sensitivity": (0.3, 0.6),
        "weekly_pattern": [0.5, 0.6, 0.7, 0.8, 1.0, 1.8, 1.6],
        "yearly_amplitude": 35,
        "yearly_peak_day": 150,
        "promo_lift": 1.6,
        "holiday_lift": 1.2,
        "return_rate": 0.03,
    },
    "Sports & Outdoors": {
        "price_range": (8, 400),
        "margin_range": (0.20, 0.50),
        "base_demand_range": (0.4, 7),
        "elasticity_range": (-2.0, -1.0),
        "promo_sensitivity": (0.3, 0.7),
        "weekly_pattern": [0.6, 0.7, 0.8, 0.9, 1.2, 1.7, 1.4],
        "yearly_amplitude": 40,
        "yearly_peak_day": 170,
        "promo_lift": 1.7,
        "holiday_lift": 1.3,
        "return_rate": 0.04,
    },
}

PROMO_TYPES = {
    "percentage_off": {"discount_range": (0.05, 0.40), "lift_range": (1.3, 2.0)},
    "buy_x_get_y": {"discount_range": (0.25, 0.50), "lift_range": (1.2, 1.6)},
    "fixed_discount": {"discount_range": (1, 20), "lift_range": (1.2, 1.8)},
    "bundle": {"discount_range": (0.10, 0.30), "lift_range": (1.2, 1.6)},
    "clearance": {"discount_range": (0.30, 0.60), "lift_range": (1.5, 2.5)},
}

US_HOLIDAYS_2022_2024 = set()
for _year in range(2022, 2025):
    US_HOLIDAYS_2022_2024.update([
        date(_year, 1, 1), date(_year, 1, 17), date(_year, 2, 21),
        date(_year, 5, 30), date(_year, 6, 19), date(_year, 7, 4),
        date(_year, 9, 5), date(_year, 10, 10), date(_year, 11, 11),
        date(_year, 11, 24), date(_year, 12, 25),
        date(_year, 11, 1), date(_year, 11, 25),
        date(_year, 12, 24), date(_year, 12, 26), date(_year, 12, 31),
    ])

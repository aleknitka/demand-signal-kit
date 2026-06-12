import polars as pl
import numpy as np
from datetime import date, timedelta

from demand_signal_kit.data.synthetic.config import SyntheticDataConfig, REGIONS


LOYALTY_TIERS = {
    "bronze": {"spend_threshold": 0, "points_multiplier": 1.0, "discount_pct": 0.0},
    "silver": {"spend_threshold": 500, "points_multiplier": 1.2, "discount_pct": 0.03},
    "gold": {"spend_threshold": 2000, "points_multiplier": 1.5, "discount_pct": 0.05},
    "platinum": {"spend_threshold": 5000, "points_multiplier": 2.0, "discount_pct": 0.10},
}

SEGMENT_PROFILES = {
    "budget": {
        "avg_basket_size": (1, 4),
        "visit_frequency_days": (7, 21),
        "price_sensitivity": (1.2, 1.8),
        "promo_response": (0.7, 1.0),
        "category偏好": {"Food & Beverage": 0.45, "Clothing": 0.15, "Home & Garden": 0.15, "Electronics": 0.10, "Sports & Outdoors": 0.15},
        "loyalty_tier_weights": {"bronze": 0.6, "silver": 0.3, "gold": 0.08, "platinum": 0.02},
    },
    "regular": {
        "avg_basket_size": (2, 7),
        "visit_frequency_days": (3, 10),
        "price_sensitivity": (0.8, 1.2),
        "promo_response": (0.4, 0.7),
        "category偏好": {"Food & Beverage": 0.30, "Clothing": 0.20, "Home & Garden": 0.20, "Electronics": 0.15, "Sports & Outdoors": 0.15},
        "loyalty_tier_weights": {"bronze": 0.2, "silver": 0.4, "gold": 0.3, "platinum": 0.1},
    },
    "premium": {
        "avg_basket_size": (3, 10),
        "visit_frequency_days": (2, 7),
        "price_sensitivity": (0.5, 0.9),
        "promo_response": (0.3, 0.6),
        "category偏好": {"Electronics": 0.25, "Clothing": 0.25, "Home & Garden": 0.20, "Food & Beverage": 0.15, "Sports & Outdoors": 0.15},
        "loyalty_tier_weights": {"bronze": 0.05, "silver": 0.15, "gold": 0.45, "platinum": 0.35},
    },
    "vip": {
        "avg_basket_size": (5, 15),
        "visit_frequency_days": (1, 5),
        "price_sensitivity": (0.3, 0.7),
        "promo_response": (0.2, 0.5),
        "category偏好": {"Electronics": 0.30, "Clothing": 0.25, "Home & Garden": 0.20, "Sports & Outdoors": 0.15, "Food & Beverage": 0.10},
        "loyalty_tier_weights": {"bronze": 0.02, "silver": 0.08, "gold": 0.30, "platinum": 0.60},
    },
    "inactive": {
        "avg_basket_size": (1, 3),
        "visit_frequency_days": (30, 90),
        "price_sensitivity": (1.0, 1.5),
        "promo_response": (0.1, 0.3),
        "category偏好": {"Food & Beverage": 0.50, "Clothing": 0.10, "Home & Garden": 0.10, "Electronics": 0.15, "Sports & Outdoors": 0.15},
        "loyalty_tier_weights": {"bronze": 0.5, "silver": 0.3, "gold": 0.15, "platinum": 0.05},
    },
}


def generate_customers(config: SyntheticDataConfig, rng: np.random.Generator) -> pl.DataFrame:
    n = config.n_customers

    seg_names = list(config.customer_segment_weights.keys())
    seg_weights = list(config.customer_segment_weights.values())
    segments = rng.choice(seg_names, size=n, p=seg_weights)

    region_names = list(REGIONS.keys())
    regions = rng.choice(region_names, size=n)

    first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
                   "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
                   "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Emma",
                   "Oliver", "Sophia", "Liam", "Charlotte", "Noah", "Amelia", "Ethan",
                   "Mia", "Lucas", "Harper", "Mason", "Evelyn", "Logan", "Abigail"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
                  "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
                  "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
                  "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
                  "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King"]

    rows = []
    for i in range(n):
        seg = segments[i]
        profile = SEGMENT_PROFILES[seg]

        first = rng.choice(first_names)
        last = rng.choice(last_names)
        region = regions[i]

        is_enrolled = bool(rng.random() < config.loyalty_enrollment_rate)
        tier_weights = list(profile["loyalty_tier_weights"].values())
        tier_names = list(profile["loyalty_tier_weights"].keys())
        tier = rng.choice(tier_names, p=tier_weights) if is_enrolled else "none"

        signup_date = date(2020, 1, 1) + timedelta(days=int(rng.integers(0, 1500)))

        email = f"{first.lower()}.{last.lower()}{i}@email.com"

        total_spend = float(rng.uniform(0, 5000)) if seg != "inactive" else float(rng.uniform(0, 200))
        total_visits = max(1, int(total_spend / rng.uniform(20, 100)))
        avg_basket = round(total_spend / total_visits, 2) if total_visits > 0 else 0

        rows.append({
            "customer_id": f"C{i+1:06d}",
            "first_name": first,
            "last_name": last,
            "email": email,
            "segment": seg,
            "region": region,
            "loyalty_enrolled": is_enrolled,
            "loyalty_tier": tier,
            "signup_date": signup_date,
            "total_spend": round(total_spend, 2),
            "total_visits": total_visits,
            "avg_basket_value": avg_basket,
            "preferred_categories": str(dict(zip(
                list(profile["category偏好"].keys()),
                [round(v, 2) for v in profile["category偏好"].values()]
            ))),
        })

    return pl.DataFrame(rows).with_columns(pl.col("signup_date").cast(pl.Date))


def get_customer_receipt_probability(
    customer: dict,
    store: dict,
    day: date,
    rng: np.random.Generator,
) -> float:
    seg = customer["segment"]
    profile = SEGMENT_PROFILES[seg]

    visit_min, visit_max = profile["visit_frequency_days"]
    avg_frequency = (visit_min + visit_max) / 2
    base_prob = 1.0 / avg_frequency

    dow = day.weekday()
    if dow >= 5:
        base_prob *= 1.3

    if seg in ("premium", "vip"):
        base_prob *= 1.2
    elif seg == "inactive":
        base_prob *= 0.3

    noise = float(rng.normal(1.0, 0.2))
    return max(0.001, min(0.5, base_prob * noise))


def get_segment_basket_size(customer: dict, rng: np.random.Generator) -> int:
    seg = customer["segment"]
    profile = SEGMENT_PROFILES[seg]
    low, high = profile["avg_basket_size"]
    return int(rng.integers(low, high + 1))


def get_segment_promo_response(customer: dict, rng: np.random.Generator) -> float:
    seg = customer["segment"]
    profile = SEGMENT_PROFILES[seg]
    low, high = profile["promo_response"]
    return float(rng.uniform(low, high))


def get_segment_price_sensitivity(customer: dict, rng: np.random.Generator) -> float:
    seg = customer["segment"]
    profile = SEGMENT_PROFILES[seg]
    low, high = profile["price_sensitivity"]
    return float(rng.uniform(low, high))

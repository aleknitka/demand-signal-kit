import polars as pl
import numpy as np
from datetime import date, timedelta


PRODUCT_PROFILES = {
    "Electronics": {
        "base_demand": 50,
        "trend": 0.02,
        "weekly_pattern": [0.8, 0.7, 0.9, 1.0, 1.2, 1.5, 1.1],
        "yearly_amplitude": 25,
        "yearly_peak_day": 340,
        "promo_lift": 1.8,
        "holiday_lift": 2.5,
    },
    "Clothing": {
        "base_demand": 80,
        "trend": 0.01,
        "weekly_pattern": [0.6, 0.7, 0.8, 0.9, 1.1, 1.6, 1.3],
        "yearly_amplitude": 30,
        "yearly_peak_day": 330,
        "promo_lift": 2.0,
        "holiday_lift": 1.5,
    },
    "Food & Beverage": {
        "base_demand": 120,
        "trend": 0.005,
        "weekly_pattern": [0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 0.8],
        "yearly_amplitude": 15,
        "yearly_peak_day": 355,
        "promo_lift": 1.4,
        "holiday_lift": 1.8,
    },
    "Home & Garden": {
        "base_demand": 40,
        "trend": 0.03,
        "weekly_pattern": [0.5, 0.6, 0.7, 0.8, 1.0, 1.8, 1.6],
        "yearly_amplitude": 35,
        "yearly_peak_day": 150,
        "promo_lift": 1.6,
        "holiday_lift": 1.2,
    },
    "Sports & Outdoors": {
        "base_demand": 35,
        "trend": 0.02,
        "weekly_pattern": [0.6, 0.7, 0.8, 0.9, 1.2, 1.7, 1.4],
        "yearly_amplitude": 40,
        "yearly_peak_day": 170,
        "promo_lift": 1.7,
        "holiday_lift": 1.3,
    },
}

US_HOLIDAYS_2022_2024 = set()

for year in range(2022, 2025):
    US_HOLIDAYS_2022_2024.update([
        date(year, 1, 1),
        date(year, 1, 17),
        date(year, 2, 21),
        date(year, 5, 30),
        date(year, 6, 19),
        date(year, 7, 4),
        date(year, 9, 5),
        date(year, 10, 10),
        date(year, 11, 11),
        date(year, 11, 24),
        date(year, 12, 25),
    ])
    US_HOLIDAYS_2022_2024.add(date(year, 11, 1))
    US_HOLIDAYS_2022_2024.add(date(year, 11, 25))
    US_HOLIDAYS_2022_2024.add(date(year, 12, 24))
    US_HOLIDAYS_2022_2024.add(date(year, 12, 26))
    US_HOLIDAYS_2022_2024.add(date(year, 12, 31))


def generate_sample_data(
    n_products: int = 5,
    start_date: date = date(2022, 1, 1),
    end_date: date = date(2024, 12, 31),
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    n_days = len(dates)

    product_names = list(PRODUCT_PROFILES.keys())[:n_products]

    rows = []
    for pid, category in enumerate(product_names, 1):
        profile = PRODUCT_PROFILES[category]
        base = profile["base_demand"]
        trend = profile["trend"]
        weekly = profile["weekly_pattern"]
        yearly_amp = profile["yearly_amplitude"]
        yearly_peak = profile["yearly_peak_day"]
        promo_lift = profile["promo_lift"]
        holiday_lift = profile["holiday_lift"]

        promo_days = set(rng.choice(n_days, size=int(n_days * 0.08), replace=False))

        for i, d in enumerate(dates):
            base_val = base + trend * i

            weekly_effect = weekly[d.weekday()] * base * 0.3

            day_of_year = d.timetuple().tm_yday
            yearly_effect = yearly_amp * np.sin(2 * np.pi * (day_of_year - yearly_peak) / 365)

            is_holiday = d in US_HOLIDAYS_2022_2024
            holiday_effect = base * 0.5 * holiday_lift if is_holiday else 0

            is_black_friday = (
                d.month == 11 and 22 <= d.day <= 30 and d.weekday() == 4
            )
            if is_black_friday:
                holiday_effect = base * 1.5

            is_promo = i in promo_days
            promo_effect = base * 0.4 * promo_lift if is_promo else 0

            noise = rng.normal(0, base * 0.08)

            qty = max(0, base_val + weekly_effect + yearly_effect + holiday_effect + promo_effect + noise)

            rows.append({
                "date": d,
                "product_id": f"P{pid:03d}",
                "category": category,
                "quantity": round(qty, 1),
                "is_holiday": int(is_holiday or is_black_friday),
                "is_promo": int(is_promo),
            })

    df = pl.DataFrame(rows)
    return df.with_columns(pl.col("date").cast(pl.Date))


def generate_demand_csv(
    output_path: str = "data/sample_demand.csv",
    **kwargs,
) -> str:
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_data(**kwargs)
    df.write_csv(output_path)
    return output_path

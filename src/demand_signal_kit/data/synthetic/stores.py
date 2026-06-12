import polars as pl
import numpy as np
from datetime import date, timedelta

from demand_signal_kit.data.synthetic.config import (
    SyntheticDataConfig,
    STORE_TYPE_SIZE_PROFILES,
    STORE_TYPE_SIZES,
    REGIONS,
    REGION_CITIES,
)


def generate_stores(config: SyntheticDataConfig, rng: np.random.Generator) -> pl.DataFrame:
    n = config.n_stores

    type_weights = list(config.store_type_weights.values())
    type_names = list(config.store_type_weights.keys())
    store_types = rng.choice(type_names, size=n, p=type_weights)

    store_sizes = []
    for st in store_types:
        sizes = STORE_TYPE_SIZES[st]
        store_sizes.append(rng.choice(sizes))

    region_names = list(REGIONS.keys())
    regions = rng.choice(region_names, size=n)

    cities = []
    states = []
    for r in regions:
        city = rng.choice(REGION_CITIES[r])
        state = rng.choice(REGIONS[r]["states"])
        cities.append(city)
        states.append(state)

    rows = []
    for i in range(n):
        st = store_types[i]
        sz = store_sizes[i]
        profile = STORE_TYPE_SIZE_PROFILES[(st, sz)]
        sqft = int(rng.integers(profile["sqft"][0], profile["sqft"][1]))
        traffic = int(rng.integers(profile["traffic"][0], profile["traffic"][1]))
        region_mod = REGIONS[regions[i]]["traffic_mod"]

        store_id = f"S{i+1:04d}"
        store_name = f"{st} {sz} #{i+1}"
        lat = float(rng.uniform(25, 48))
        lon = float(rng.uniform(-125, -70))
        open_date = date(2018, 1, 1) + timedelta(days=int(rng.integers(0, 1800)))
        price_sens = float(rng.uniform(0.5, 1.5))
        promo_resp = float(rng.uniform(0.3, 1.8))

        rows.append({
            "store_id": store_id,
            "store_name": store_name,
            "store_type": st,
            "store_size": sz,
            "region": regions[i],
            "city": cities[i],
            "state": states[i],
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "open_date": open_date,
            "sqft": sqft,
            "weekly_foot_traffic": round(traffic * region_mod, 0),
            "price_sensitivity": round(price_sens, 3),
            "promo_response_rate": round(promo_resp, 3),
        })

    return pl.DataFrame(rows).with_columns(pl.col("open_date").cast(pl.Date))

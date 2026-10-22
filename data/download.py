"""
Data Ingestion and Temporal Dataset Preparation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NYC_TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"


def generate_synthetic_taxi_fallback(
    n_samples_per_window: int = 15000,
    n_windows: int = 6,
    random_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Generates realistic temporal taxi datasets with engineered real-world drift scenarios
    using an isolated local random generator (no global np.random state side effects).
    """
    rng = np.random.default_rng(random_seed)
    windows: Dict[str, pd.DataFrame] = {}

    base_start = pd.Timestamp("2023-01-01")

    for w_idx in range(n_windows):
        w_name = f"W{w_idx}_{'reference' if w_idx == 0 else f'prod_month_{w_idx}'}"
        base_time = base_start + pd.DateOffset(months=w_idx)

        seconds_offset = rng.uniform(0, 28 * 86400, size=n_samples_per_window)
        pickup_times = pd.Series(base_time + pd.to_timedelta(seconds_offset, unit="s"))
        hour = pickup_times.dt.hour.to_numpy()
        day_of_week = pickup_times.dt.dayofweek.to_numpy()
        is_weekend = (day_of_week >= 5).astype(int)

        dist_mean, log_sigma = 2.8, 0.60
        passengers = rng.choice([1, 2, 3, 4, 5, 6], size=n_samples_per_window, p=[0.70, 0.15, 0.05, 0.04, 0.04, 0.02])
        payment_type = rng.choice([1, 2, 3, 4], size=n_samples_per_window, p=[0.78, 0.20, 0.01, 0.01])
        ratecode = rng.choice([1, 2, 3, 4, 5], size=n_samples_per_window, p=[0.94, 0.03, 0.01, 0.01, 0.01])
        pu_location = rng.choice(range(1, 264), size=n_samples_per_window)
        do_location = rng.choice(range(1, 264), size=n_samples_per_window)

        if w_idx == 1:
            pass
        elif w_idx == 2:
            dist_mean += 1.8
            log_sigma = 0.72
            ratecode = rng.choice([1, 2, 3, 4, 5], size=n_samples_per_window, p=[0.82, 0.12, 0.03, 0.02, 0.01])
        elif w_idx == 3:
            dist_mean += 0.2
        elif w_idx == 4:
            dist_mean = max(1.2, dist_mean - 1.0)
            passengers = rng.choice([1, 2, 3, 4, 5, 6], size=n_samples_per_window, p=[0.85, 0.08, 0.03, 0.02, 0.01, 0.01])
        elif w_idx >= 5:
            payment_type = rng.choice([1, 2, 3, 4], size=n_samples_per_window, p=[0.92, 0.07, 0.005, 0.005])

        trip_distance = rng.lognormal(mean=np.log(dist_mean + 0.1), sigma=log_sigma, size=n_samples_per_window)
        trip_distance = np.clip(trip_distance, 0.2, 35.0)

        rush_hour_factor = np.where((hour >= 7) & (hour <= 9) | (hour >= 16) & (hour <= 19), 1.45, 1.0)
        speed_mph = rng.uniform(9.0, 22.0, size=n_samples_per_window) / rush_hour_factor
        if w_idx == 4:
            speed_mph *= 0.65
        trip_duration_min = (trip_distance / speed_mph) * 60.0 + rng.exponential(2.0, size=n_samples_per_window)
        trip_duration_min = np.clip(trip_duration_min, 1.0, 120.0)

        dropoff_times = pickup_times + pd.to_timedelta(trip_duration_min, unit="m")

        base_fare = 3.00
        per_mile = 2.00 if w_idx >= 3 else 1.75
        fare_amount = base_fare + (trip_distance * per_mile) + (trip_duration_min * 0.40)
        fare_amount += rng.normal(0, 1.0, size=n_samples_per_window)
        fare_amount = np.clip(fare_amount, 3.50, 250.0)

        congestion_surcharge = np.where(pu_location < 200, 2.50, 0.0)
        if w_idx >= 3:
            congestion_surcharge *= 1.2
        tolls_amount = np.where(
            trip_distance > 8.0,
            rng.choice([0.0, 6.55, 13.75], size=n_samples_per_window, p=[0.3, 0.5, 0.2]),
            0.0,
        )
        extra = np.where((hour >= 20) | (hour < 6), 1.00, 0.0)

        is_credit = (payment_type == 1).astype(float)
        base_tip_pct = 0.18 + (0.04 if w_idx >= 5 else 0.0)
        tip_amount = np.where(
            is_credit == 1,
            np.maximum(0.0, fare_amount * (base_tip_pct + rng.normal(0, 0.05, size=n_samples_per_window))),
            0.0,
        )
        total_amount = fare_amount + extra + tolls_amount + congestion_surcharge + tip_amount

        df = pd.DataFrame({
            "tpep_pickup_datetime": pickup_times,
            "tpep_dropoff_datetime": dropoff_times,
            "passenger_count": passengers,
            "trip_distance": np.round(trip_distance, 2),
            "PULocationID": pu_location,
            "DOLocationID": do_location,
            "RatecodeID": ratecode,
            "payment_type": payment_type,
            "fare_amount": np.round(fare_amount, 2),
            "extra": np.round(extra, 2),
            "mta_tax": 0.50,
            "tip_amount": np.round(tip_amount, 2),
            "tolls_amount": np.round(tolls_amount, 2),
            "improvement_surcharge": 1.00,
            "total_amount": np.round(total_amount, 2),
            "congestion_surcharge": np.round(congestion_surcharge, 2),
            "trip_duration_min": np.round(trip_duration_min, 2),
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
        })
        windows[w_name] = df
        logger.info("Generated synthetic window %s with %d records.", w_name, len(df))

    return windows
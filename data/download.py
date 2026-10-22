"""
Data Ingestion and Temporal Dataset Preparation.

Dataset Selection Justification:
--------------------------------
We use the New York City Taxi and Limousine Commission (NYC TLC) Trip Record Dataset.
1. Natural Temporal Dynamics: Taxi mobility data has rich temporal hierarchies
   (hourly peak commuting, weekday vs weekend patterns, seasonal weather shifts,
   macroeconomic inflation in base fares, and fuel/congestion surcharges).
2. Complex Tabular Interactions: Contains both continuous physical variables
   (trip_distance, fare_amount, speed) and high-cardinality discrete variables
   (PULocationID, DOLocationID, RatecodeID, payment_type).
3. The "Silent Drift" / Delayed Label Problem: In real fleet and ride-hailing deployments,
   ground truth targets (e.g. final tip amount, customer satisfaction rating, dispute/refund
   status) are delayed by days or weeks, whereas operational features are logged instantaneously.
   This makes covariate shift detection without ground truth vital.

This module provides both live streaming/fetching of official NYC TLC Parquet files and
an offline statistical generative engine matching the empirical joint distributions.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    Generates realistic temporal taxi datasets with engineered real-world drift scenarios.
    
    Statistical Properties Modeled:
    - W0 (Reference / Baseline): Standard urban mobility distribution.
    - W1 (Production - Stable): Normal random variation, no systematic drift (PSI < 0.05).
    - W2 (Production - Airport/Distance Shift): Longer highway trips, shifted trip_distance & toll.
    - W3 (Production - Inflation/Congestion Shift): Fare rate increase, higher congestion surcharge.
    - W4 (Production - Severe Multi-Feature Shift): Seasonal weather disruption + route shifts.
    - W5 (Production - Payment / Tip Behavior Shift): Shift in credit card usage and tipping culture.
    """
    np.random.seed(random_seed)
    windows: Dict[str, pd.DataFrame] = {}

    start_dates = [
        "2023-01-01", "2023-02-01", "2023-03-01", 
        "2023-04-01", "2023-05-01", "2023-06-01"
    ]

    for w_idx in range(n_windows):
        w_name = f"W{w_idx}_{'reference' if w_idx == 0 else f'prod_month_{w_idx}'}"
        base_time = pd.Timestamp(start_dates[w_idx])

        # Temporal timestamps over 30 days
        seconds_offset = np.random.uniform(0, 30 * 86400, size=n_samples_per_window)
        pickup_times = pd.Series(base_time + pd.to_timedelta(seconds_offset, unit="s"))
        hour = pickup_times.dt.hour.to_numpy()
        day_of_week = pickup_times.dt.dayofweek.to_numpy()
        is_weekend = (day_of_week >= 5).astype(int)

        # Baseline parameters
        dist_mean, dist_sigma = 2.8, 2.5
        passengers = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples_per_window, p=[0.70, 0.15, 0.05, 0.04, 0.04, 0.02])
        payment_type = np.random.choice([1, 2, 3, 4], size=n_samples_per_window, p=[0.78, 0.20, 0.01, 0.01])
        ratecode = np.random.choice([1, 2, 3, 4, 5], size=n_samples_per_window, p=[0.94, 0.03, 0.01, 0.01, 0.01])
        pu_location = np.random.choice(range(1, 264), size=n_samples_per_window)
        do_location = np.random.choice(range(1, 264), size=n_samples_per_window)

        # Inject realistic macro-temporal shifts per window
        if w_idx == 1:
            # Stable window: minor stochastic fluctuation
            pass
        elif w_idx == 2:
            # Shift in trip distance (e.g. airport travel season)
            dist_mean += 1.8
            dist_sigma += 0.8
            ratecode = np.random.choice([1, 2, 3, 4, 5], size=n_samples_per_window, p=[0.82, 0.12, 0.03, 0.02, 0.01])
        elif w_idx == 3:
            # Congestion & Fare inflation shift
            dist_mean += 0.2
        elif w_idx == 4:
            # Severe storm / weather disruption: short, congested trips, low speed
            dist_mean = max(1.2, dist_mean - 1.0)
            passengers = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples_per_window, p=[0.85, 0.08, 0.03, 0.02, 0.01, 0.01])
        elif w_idx == 5:
            # Payment behavior & tipping culture shift
            payment_type = np.random.choice([1, 2, 3, 4], size=n_samples_per_window, p=[0.92, 0.07, 0.005, 0.005])

        # Generate trip distance (strictly positive log-normal mixture)
        trip_distance = np.random.lognormal(mean=np.log(dist_mean + 0.1), sigma=0.6, size=n_samples_per_window)
        trip_distance = np.clip(trip_distance, 0.2, 35.0)

        # Trip duration (minutes) driven by distance, congestion, and hour
        rush_hour_factor = np.where((hour >= 7) & (hour <= 9) | (hour >= 16) & (hour <= 19), 1.45, 1.0)
        speed_mph = np.random.uniform(9.0, 22.0, size=n_samples_per_window) / rush_hour_factor
        if w_idx == 4:
            speed_mph *= 0.65  # severe weather congestion
        trip_duration_min = (trip_distance / speed_mph) * 60.0 + np.random.exponential(2.0, size=n_samples_per_window)
        trip_duration_min = np.clip(trip_duration_min, 1.0, 120.0)

        dropoff_times = pickup_times + pd.to_timedelta(trip_duration_min, unit="m")

        # Fare calculation based on NYC TLC rate structure ($3.00 base + $1.75/mile + time)
        base_fare = 3.00
        per_mile = 2.00 if w_idx >= 3 else 1.75
        fare_amount = base_fare + (trip_distance * per_mile) + (trip_duration_min * 0.40)
        fare_amount += np.random.normal(0, 1.0, size=n_samples_per_window)
        fare_amount = np.clip(fare_amount, 3.50, 250.0)

        # Extra charges (Congestion surcharge $2.50, airport access $1.25, tolls)
        congestion_surcharge = np.where(pu_location < 200, 2.50, 0.0)
        if w_idx >= 3:
            congestion_surcharge *= 1.2  # congestion price hike
        tolls_amount = np.where(trip_distance > 8.0, np.random.choice([0.0, 6.55, 13.75], size=n_samples_per_window, p=[0.3, 0.5, 0.2]), 0.0)
        extra = np.where((hour >= 20) | (hour < 6), 1.00, 0.0)

        # Target: Tip Amount ($)
        # Tip depends on payment_type (cash = 0 tip recorded in TLC), fare_amount, distance, service quality proxy
        is_credit = (payment_type == 1).astype(float)
        base_tip_pct = 0.18 + (0.04 if w_idx == 5 else 0.0)
        tip_amount = np.where(
            is_credit == 1,
            np.maximum(0.0, fare_amount * (base_tip_pct + np.random.normal(0, 0.05, size=n_samples_per_window))),
            0.0
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


def fetch_taxi_data(
    raw_dir: str = "./data/raw",
    year: int = 2023,
    months: Optional[List[int]] = None,
    samples_per_month: int = 25000,
    force_synthetic: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Downloads official NYC TLC trip parquet files or loads locally cached partitions.
    Falls back to high-fidelity synthetic generator if offline.
    """
    if months is None:
        months = [1, 2, 3, 4, 5, 6]

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    datasets: Dict[str, pd.DataFrame] = {}

    if force_synthetic:
        logger.info("Generating realistic temporal taxi partitions...")
        return generate_synthetic_taxi_fallback(n_samples_per_window=samples_per_month, n_windows=len(months))

    try:
        import requests
        for idx, month in enumerate(months):
            w_name = f"W{idx}_{'reference' if idx == 0 else f'prod_month_{month:02d}'}"
            file_name = f"yellow_tripdata_{year}-{month:02d}.parquet"
            local_file = raw_path / file_name

            if not local_file.exists():
                url = NYC_TLC_BASE_URL.format(year=year, month=month)
                logger.info("Downloading NYC TLC data from %s...", url)
                resp = requests.get(url, timeout=30, stream=True)
                if resp.status_code == 200:
                    with open(local_file, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            f.write(chunk)
                    logger.info("Saved %s", local_file)
                else:
                    logger.warning("Failed to download from %s (Status: %d). Falling back to synthetic generator.", url, resp.status_code)
                    return generate_synthetic_taxi_fallback(n_samples_per_window=samples_per_month, n_windows=len(months))

            # Load parquet
            df = pd.read_parquet(local_file)
            if len(df) > samples_per_month:
                df = df.sample(n=samples_per_month, random_state=42 + idx).reset_index(drop=True)
            
            datasets[w_name] = df
            logger.info("Loaded window %s: %d records.", w_name, len(df))

        return datasets

    except Exception as e:
        logger.warning("Encountered error fetching remote parquet (%s). Falling back to statistical generator.", str(e))
        return generate_synthetic_taxi_fallback(n_samples_per_window=samples_per_month, n_windows=len(months))

"""
Data preprocessing, feature engineering, and dataset splitting pipeline.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Core tabular feature columns used across detection, modeling, and attribution
NUMERICAL_FEATURES = [
    "trip_distance",
    "trip_duration_min",
    "fare_amount",
    "speed_mph",
    "extra",
    "tolls_amount",
    "congestion_surcharge",
    "fare_per_mile",
]

CATEGORICAL_FEATURES = [
    "passenger_count",
    "payment_type",
    "RatecodeID",
    "hour",
    "day_of_week",
    "is_weekend",
    "PU_density_bucket",
    "DO_density_bucket",
]

TARGET_COLUMN = "tip_amount"
ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


class TaxiFeaturePipeline:
    """
    Stateful feature transformation pipeline fitted exclusively on the Reference Window (W0).
    Ensures zero data leakage from future production periods.
    """

    def __init__(self, top_k_locations: int = 15):
        self.top_k_locations = top_k_locations
        self.pu_top_categories: List[int] = []
        self.do_top_categories: List[int] = []
        self.feature_medians: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "TaxiFeaturePipeline":
        """Fits location distributions and baseline imputation medians on reference window."""
        df_clean = self._clean_raw(df)
        
        # High cardinality location density binning
        if "PULocationID" in df_clean.columns:
            pu_counts = df_clean["PULocationID"].value_counts()
            self.pu_top_categories = pu_counts.head(self.top_k_locations).index.tolist()
            
        if "DOLocationID" in df_clean.columns:
            do_counts = df_clean["DOLocationID"].value_counts()
            self.do_top_categories = do_counts.head(self.top_k_locations).index.tolist()

        # Compute numerical medians for imputation
        for col in NUMERICAL_FEATURES:
            if col in df_clean.columns:
                self.feature_medians[col] = float(df_clean[col].median())

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw dataframe into model-ready features."""
        df_transformed = self._clean_raw(df).copy()

        # Density bucket encoding
        if "PULocationID" in df_transformed.columns:
            df_transformed["PU_density_bucket"] = df_transformed["PULocationID"].apply(
                lambda x: x if x in self.pu_top_categories else -1
            )
        else:
            df_transformed["PU_density_bucket"] = -1

        if "DOLocationID" in df_transformed.columns:
            df_transformed["DO_density_bucket"] = df_transformed["DOLocationID"].apply(
                lambda x: x if x in self.do_top_categories else -1
            )
        else:
            df_transformed["DO_density_bucket"] = -1

        # Impute missing values with reference medians
        for col, med in self.feature_medians.items():
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].fillna(med)

        # Categorical columns
        for col in CATEGORICAL_FEATURES:
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].fillna(-1).astype(int)

        # Ensure all model features exist
        for col in ALL_MODEL_FEATURES:
            if col not in df_transformed.columns:
                df_transformed[col] = 0.0

        return df_transformed

    def _clean_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives clean duration, speeds, temporal indicators, and clips realistic physical bounds."""
        res = df.copy()

        # Datetime derivation
        if "tpep_pickup_datetime" in res.columns:
            res["pickup_dt"] = pd.to_datetime(res["tpep_pickup_datetime"])
            res["hour"] = res["pickup_dt"].dt.hour
            res["day_of_week"] = res["pickup_dt"].dt.dayofweek
            res["is_weekend"] = (res["day_of_week"] >= 5).astype(int)

        if "tpep_dropoff_datetime" in res.columns and "tpep_pickup_datetime" in res.columns:
            res["dropoff_dt"] = pd.to_datetime(res["tpep_dropoff_datetime"])
            duration_min = (res["dropoff_dt"] - res["pickup_dt"]).dt.total_seconds() / 60.0
            res["trip_duration_min"] = np.clip(duration_min, 1.0, 180.0)

        if "trip_duration_min" not in res.columns:
            res["trip_duration_min"] = 15.0

        # Physical speed calculation (mph)
        if "trip_distance" in res.columns and "trip_duration_min" in res.columns:
            speed = res["trip_distance"] / (res["trip_duration_min"] / 60.0 + 1e-4)
            res["speed_mph"] = np.clip(speed, 0.5, 75.0)

        # Fare per mile
        if "fare_amount" in res.columns and "trip_distance" in res.columns:
            res["fare_per_mile"] = res["fare_amount"] / (res["trip_distance"] + 0.1)
            res["fare_per_mile"] = np.clip(res["fare_per_mile"], 0.5, 30.0)

        # Fill default columns if not present
        defaults = {
            "extra": 0.0,
            "tolls_amount": 0.0,
            "congestion_surcharge": 0.0,
            "passenger_count": 1,
            "payment_type": 1,
            "RatecodeID": 1,
            "fare_amount": 10.0,
            "trip_distance": 2.5,
            "tip_amount": 2.0,
        }
        for k, v in defaults.items():
            if k not in res.columns:
                res[k] = v

        return res


def prepare_drift_datasets(
    raw_windows: Dict[str, pd.DataFrame]
) -> Tuple[TaxiFeaturePipeline, Dict[str, pd.DataFrame]]:
    """
    Fits the feature pipeline on Reference window W0, and transforms all production windows.
    Returns the fitted pipeline and dictionary of model-ready DataFrames.
    """
    w_keys = sorted(list(raw_windows.keys()))
    ref_key = w_keys[0]
    ref_df = raw_windows[ref_key]

    logger.info("Fitting TaxiFeaturePipeline on reference window %s (%d records)...", ref_key, len(ref_df))
    pipeline = TaxiFeaturePipeline().fit(ref_df)

    processed_windows: Dict[str, pd.DataFrame] = {}
    for k in w_keys:
        processed_windows[k] = pipeline.transform(raw_windows[k])
        logger.info("Processed window %s -> shape %s", k, processed_windows[k].shape)

    return pipeline, processed_windows

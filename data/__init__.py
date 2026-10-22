"""
Data ingestion, preprocessing, and baseline modeling module.
Provides real-world temporal NYC TLC taxi data loading with offline fallback.
"""
from .download import fetch_taxi_data, generate_synthetic_taxi_fallback
from .preprocess import prepare_drift_datasets, TaxiFeaturePipeline
from .model import BaselineTaxiModel, train_baseline_model

__all__ = [
    "fetch_taxi_data",
    "generate_synthetic_taxi_fallback",
    "prepare_drift_datasets",
    "TaxiFeaturePipeline",
    "BaselineTaxiModel",
    "train_baseline_model",
]

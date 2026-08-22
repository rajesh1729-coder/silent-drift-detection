"""
Unit tests for Synthetic Drift Injector and Benchmark Evaluation Suite.
"""

import numpy as np
import pandas as pd
import pytest

from data.download import generate_synthetic_taxi_fallback
from data.preprocess import prepare_drift_datasets
from data.model import BaselineTaxiModel
from detection.detector import DriftDetector
from attribution.ranker import DriftAttributionEngine
from validation.drift_injector import SyntheticDriftInjector, DriftType
from validation.benchmark import AttributionBenchmarkRunner


@pytest.fixture(scope="module")
def test_env():
    raw_windows = generate_synthetic_taxi_fallback(n_samples_per_window=2000, n_windows=2)
    pipeline, proc_windows = prepare_drift_datasets(raw_windows)
    ref_df = proc_windows["W0_reference"]

    model = BaselineTaxiModel(n_estimators=30, max_depth=4)
    model.fit(ref_df)

    detector = DriftDetector(
        numerical_features=["trip_distance", "fare_amount", "speed_mph", "trip_duration_min"],
        categorical_features=["passenger_count", "payment_type"],
    )
    detector.fit_reference(ref_df)

    attributor = DriftAttributionEngine(model, max_samples=500)
    attributor.fit_reference(ref_df)

    return model, detector, attributor, ref_df


def test_drift_injector_mean_shift(test_env):
    """
    Verifies that injecting mean shift increases the feature mean by the specified magnitude.
    """
    _, _, _, ref_df = test_env
    injector = SyntheticDriftInjector(random_seed=42)
    scenario = injector.inject_drift(ref_df, target_feature="trip_distance", drift_type=DriftType.MEAN_SHIFT, magnitude=3.0)

    clean_mean = float(ref_df["trip_distance"].mean())
    clean_std = float(ref_df["trip_distance"].std())
    drifted_mean = float(scenario.drifted_df["trip_distance"].mean())

    assert drifted_mean > clean_mean + (2.5 * clean_std)


def test_benchmark_runner_identifies_injected_root_cause(test_env):
    """
    Verifies that the benchmark runner correctly tracks precision and identifies injected drift.
    """
    model, detector, attributor, ref_df = test_env
    runner = AttributionBenchmarkRunner(model, detector, attributor)

    # Test small grid with clear distinct shift
    clean_heldout = ref_df.sample(500, random_state=42).reset_index(drop=True)
    bench_res = runner.run_benchmark_grid(
        heldout_clean_df=clean_heldout,
        features_to_test=["fare_amount", "trip_distance"],
        magnitudes=[3.0],
        drift_types=[DriftType.MEAN_SHIFT],
    )

    assert bench_res.total_experiments == 2
    assert bench_res.precision_at_1_unified >= 0.50
    assert bench_res.recall_at_3_unified == 1.0
    assert bench_res.mrr_unified > 0.60

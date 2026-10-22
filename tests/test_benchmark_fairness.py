"""Tests that the benchmark compares methods fairly and does not leak labels."""

import pytest

from data.download import generate_synthetic_taxi_fallback
from data.model import BaselineTaxiModel
from data.preprocess import prepare_drift_datasets
from detection.detector import DriftDetector
from attribution.ranker import DriftAttributionEngine
from validation.benchmark import AttributionBenchmarkRunner, make_fresh_heldout
from validation.drift_injector import DriftType


@pytest.fixture(scope="module")
def env():
    raw = generate_synthetic_taxi_fallback(n_samples_per_window=2000, n_windows=2)
    pipeline, proc = prepare_drift_datasets(raw)
    ref_df = proc["W0_reference"]
    model = BaselineTaxiModel(n_estimators=30, max_depth=4)
    model.fit(ref_df)
    detector = DriftDetector(
        numerical_features=["trip_distance", "fare_amount", "speed_mph", "trip_duration_min"],
        categorical_features=["passenger_count", "payment_type"],
    )
    detector.fit_reference(ref_df)
    attributor = DriftAttributionEngine(model, max_samples=500)
    attributor.fit_reference(ref_df)
    return pipeline, model, detector, attributor, ref_df


def _small_grid(runner, clean, **kwargs):
    return runner.run_benchmark_grid(
        heldout_clean_df=clean,
        features_to_test=["fare_amount", "trip_distance"],
        magnitudes=[3.0],
        drift_types=[DriftType.MEAN_SHIFT],
        **kwargs,
    )


def _spy_calls(monkeypatch, attributor):
    calls = []
    original = attributor.attribute_drift

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(attributor, "attribute_drift", spy)
    return calls


def test_default_benchmark_hides_labels_and_detector_shortlist(env, monkeypatch):
    pipeline, model, detector, attributor, ref_df = env
    calls = _spy_calls(monkeypatch, attributor)
    runner = AttributionBenchmarkRunner(model, detector, attributor)
    _small_grid(runner, ref_df.sample(500, random_state=1).reset_index(drop=True))
    assert len(calls) == 2
    assert all(c["target_col"] is None for c in calls)
    assert all(c["drift_report"] is None for c in calls)


def test_original_setup_can_still_be_reproduced(env, monkeypatch):
    pipeline, model, detector, attributor, ref_df = env
    calls = _spy_calls(monkeypatch, attributor)
    runner = AttributionBenchmarkRunner(model, detector, attributor)
    _small_grid(
        runner,
        ref_df.sample(500, random_state=1).reset_index(drop=True),
        use_labels=True,
        rank_all_features=False,
    )
    assert all(c["target_col"] == model.target_col for c in calls)
    assert all(c["drift_report"] is not None for c in calls)


def test_all_methods_rank_the_same_full_feature_list(env):
    pipeline, model, detector, attributor, ref_df = env
    runner = AttributionBenchmarkRunner(model, detector, attributor)
    res = _small_grid(runner, ref_df.sample(500, random_state=1).reset_index(drop=True))
    n_features = len(model.features)
    for _, row in res.experiment_records.iterrows():
        assert row["rank_unified"] <= n_features
        assert row["rank_shap"] <= n_features
        assert row["rank_causal"] <= n_features
    causal_lists = [len(attributor.attribute_drift(ref_df.head(300), drift_report=None).causal_ablation.ranked_causal_features)]
    assert causal_lists[0] == n_features


def test_psi_baseline_is_reported(env):
    pipeline, model, detector, attributor, ref_df = env
    runner = AttributionBenchmarkRunner(model, detector, attributor)
    res = _small_grid(runner, ref_df.sample(500, random_state=1).reset_index(drop=True))
    assert 0.0 <= res.precision_at_1_psi <= 1.0
    assert 0.0 <= res.mrr_psi <= 1.0
    assert "is_top1_correct_psi" in res.experiment_records.columns


def test_fresh_heldout_is_not_a_copy_of_training_data(env):
    pipeline, model, detector, attributor, ref_df = env
    fresh = make_fresh_heldout(pipeline, n_samples=1000, random_seed=777)
    assert len(fresh) == 1000
    assert set(model.features).issubset(fresh.columns)
    assert fresh["trip_distance"].head(50).tolist() != ref_df["trip_distance"].head(50).tolist()


def test_no_drift_control_reports_false_alarm_rate(env):
    pipeline, model, detector, attributor, ref_df = env
    runner = AttributionBenchmarkRunner(model, detector, attributor)
    fresh = make_fresh_heldout(pipeline, n_samples=1500, random_seed=777)
    out = runner.run_no_drift_control(fresh, n_chunks=3, chunk_size=800)
    assert out["n_chunks"] == 3.0
    assert 0.0 <= out["false_alarm_rate"] <= 1.0
    assert out["false_critical_rate"] <= out["false_alarm_rate"]

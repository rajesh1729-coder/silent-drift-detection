"""
Script to train the baseline model, run the synthetic validation benchmark,
and generate high-resolution visualization assets for README and docs.
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from data.download import fetch_taxi_data
from data.preprocess import prepare_drift_datasets, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from data.model import BaselineTaxiModel
from detection.detector import DriftDetector
from attribution.ranker import DriftAttributionEngine
from validation.drift_injector import SyntheticDriftInjector, DriftType
from validation.benchmark import (
    AttributionBenchmarkRunner,
    generate_benchmark_visualizations,
    make_fresh_heldout,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    assets_dir = Path("./assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== STEP 1: Ingesting Temporal Dataset ===")
    raw_windows = fetch_taxi_data(samples_per_month=15000, force_synthetic=True)
    pipeline, proc_windows = prepare_drift_datasets(raw_windows)

    ref_key = list(proc_windows.keys())[0]
    ref_df = proc_windows[ref_key]

    logger.info("=== STEP 2: Training Production Baseline Model ===")
    model = BaselineTaxiModel()
    model.fit(ref_df)
    model.save("./models/baseline_gb_model.joblib")

    logger.info("=== STEP 3: Initializing Multi-Signal Detector & Attribution Engine ===")
    detector = DriftDetector(
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    ref_preds = model.predict(ref_df)
    ref_uncs = model.estimate_uncertainty(ref_df)
    detector.fit_reference(ref_df, ref_preds, ref_uncs)

    attributor = DriftAttributionEngine(model_wrapper=model, max_samples=1500)
    attributor.fit_reference(ref_df)

    logger.info("=== STEP 4: Running Comprehensive Synthetic Validation Suite (Unseen Held-Out Split) ===")
    clean_val = make_fresh_heldout(pipeline, n_samples=4000, random_seed=777)

    runner = AttributionBenchmarkRunner(model, detector, attributor)
    bench_result = runner.run_benchmark_grid(
        heldout_clean_df=clean_val,
        features_to_test=[
            "trip_distance",
            "fare_amount",
            "speed_mph",
            "trip_duration_min",
            "extra",
            "congestion_surcharge",
            "payment_type",
            "passenger_count",
        ],
        magnitudes=[0.75, 1.5, 2.5, 4.0],
        drift_types=[DriftType.MEAN_SHIFT, DriftType.VARIANCE_SCALE, DriftType.NONLINEAR_TRANSFORM],
        use_labels=False,
        rank_all_features=True,
    )

    logger.info("=== STEP 5: Generating High-Resolution Assets ===")
    generate_benchmark_visualizations(bench_result, save_dir=str(assets_dir))

    bench_result.experiment_records.to_csv(assets_dir / "benchmark_experiment_records.csv", index=False)
    bench_result.summary_by_feature.to_csv(assets_dir / "summary_by_feature.csv", index=False)
    bench_result.summary_by_magnitude.to_csv(assets_dir / "summary_by_magnitude.csv", index=False)

    logger.info("Generating Case Study visualization for trip_distance mean shift...")
    injector = SyntheticDriftInjector(random_seed=42)
    case_scenario = injector.inject_drift(
        clean_df=clean_val,
        target_feature="trip_distance",
        drift_type=DriftType.MEAN_SHIFT,
        magnitude=2.5,
    )
    case_preds = model.predict(case_scenario.drifted_df)
    case_uncs = model.estimate_uncertainty(case_scenario.drifted_df)
    case_rep = detector.inspect_window(
        case_scenario.drifted_df,
        window_name="Case_Study_Trip_Distance",
        prod_predictions=case_preds,
        prod_uncertainties=case_uncs,
    )
    case_attr = attributor.attribute_drift(case_scenario.drifted_df, drift_report=case_rep, target_col=None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    shap_df = (
        pd.DataFrame(list(case_attr.shap_attribution.delta_mean_shaps.items()), columns=["Feature", "Delta_SHAP"])
        .sort_values(by="Delta_SHAP", key=abs, ascending=True)
        .tail(8)
    )
    colors = ["#d62728" if f == "trip_distance" else "#1f77b4" for f in shap_df["Feature"]]
    ax1.barh(shap_df["Feature"], shap_df["Delta_SHAP"], color=colors)
    ax1.set_title("SHAP Mean Shift Decomposition (Δ E[φ_j])", weight="bold")
    ax1.set_xlabel("Mean SHAP Shift ($)")

    causal_df = (
        pd.DataFrame(
            list(case_attr.causal_ablation.prediction_recovery_scores.items()), columns=["Feature", "Recovery"]
        )
        .sort_values(by="Recovery", ascending=True)
        .tail(8)
    )
    causal_df["Recovery_Pct"] = causal_df["Recovery"] * 100.0
    c_colors = ["#2ca02c" if f == "trip_distance" else "#7f7f7f" for f in causal_df["Feature"]]
    ax2.barh(causal_df["Feature"], causal_df["Recovery_Pct"], color=c_colors)
    ax2.set_title("Causal Ablation: Prediction Gap Closure (%)", weight="bold")
    ax2.set_xlabel("Recovery (%)")

    plt.tight_layout()
    case_path = assets_dir / "attribution_case_study.png"
    fig.savefig(case_path, dpi=200)
    plt.close(fig)
    logger.info("Saved case study plot to %s", case_path)

    print("\n" + "=" * 50)
    print("FINAL BENCHMARK ACCURACY SUMMARY")
    print("=" * 50)
    print(f"Total Experiments:          {bench_result.total_experiments}")
    print(f"Unified Engine Precision@1: {bench_result.precision_at_1_unified * 100:.1f}%")
    print(f"Unified Engine Recall@3:    {bench_result.recall_at_3_unified * 100:.1f}%")
    print(f"Unified Engine MRR:         {bench_result.mrr_unified:.4f}")
    print(f"SHAP-Only Precision@1:      {bench_result.precision_at_1_shap * 100:.1f}%")
    print(f"Causal-Only Precision@1:    {bench_result.precision_at_1_causal * 100:.1f}%")
    print(f"PSI-Baseline Precision@1:   {bench_result.precision_at_1_psi * 100:.1f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
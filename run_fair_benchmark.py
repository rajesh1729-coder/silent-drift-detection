"""
Runs the fair attribution benchmark and prints the results.

Fair means: truly unseen clean data, no labels given to attribution, every method
ranks all features, a plain PSI-ranking baseline is included, and a no-drift control
measures false alarms.

Usage:
    python run_fair_benchmark.py
"""

import logging
from pathlib import Path

from data.download import generate_synthetic_taxi_fallback
from data.model import BaselineTaxiModel
from data.preprocess import prepare_drift_datasets, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from detection.detector import DriftDetector
from attribution.ranker import DriftAttributionEngine
from validation.benchmark import AttributionBenchmarkRunner, make_fresh_heldout

logging.getLogger().setLevel(logging.WARNING)


def main() -> None:
    raw = generate_synthetic_taxi_fallback(n_samples_per_window=15000, n_windows=6, random_seed=42)
    pipeline, proc = prepare_drift_datasets(raw)
    ref_df = proc["W0_reference"]

    model = BaselineTaxiModel()
    model.fit(ref_df)

    detector = DriftDetector(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    detector.fit_reference(ref_df, model.predict(ref_df), model.estimate_uncertainty(ref_df))

    attributor = DriftAttributionEngine(model_wrapper=model, max_samples=1500)
    attributor.fit_reference(ref_df)

    fresh = make_fresh_heldout(pipeline, n_samples=4000, random_seed=777)
    runner = AttributionBenchmarkRunner(model, detector, attributor)
    result = runner.run_benchmark_grid(fresh)

    print(f"\nFair benchmark, N = {result.total_experiments} injected-drift experiments")
    print(f"{'method':<22}{'Precision@1':>12}{'Recall@3':>10}{'MRR':>8}")
    rows = [
        ("Unified engine", result.precision_at_1_unified, result.recall_at_3_unified, result.mrr_unified),
        ("SHAP only", result.precision_at_1_shap, result.recall_at_3_shap, result.mrr_shap),
        ("Counterfactual only", result.precision_at_1_causal, result.recall_at_3_causal, result.mrr_causal),
        ("PSI ranking baseline", result.precision_at_1_psi, result.recall_at_3_psi, result.mrr_psi),
    ]
    for name, p1, r3, mrr in rows:
        print(f"{name:<22}{p1 * 100:>11.1f}%{r3 * 100:>9.1f}%{mrr:>8.3f}")

    per_feature = result.experiment_records.groupby("target_feature")[
        ["is_top1_correct_unified", "is_top1_correct_shap", "is_top1_correct_causal", "is_top1_correct_psi"]
    ].mean().round(2)
    per_feature.columns = ["unified", "shap", "counterfactual", "psi_baseline"]
    print("\nPrecision@1 by injected feature:")
    print(per_feature.to_string())

    control = runner.run_no_drift_control(fresh, n_chunks=10, chunk_size=1500)
    print(
        f"\nNo-drift control: {control['false_alarm_rate'] * 100:.0f}% of "
        f"{int(control['n_chunks'])} clean chunks raised an alert "
        f"({control['false_critical_rate'] * 100:.0f}% critical)."
    )

    out_dir = Path("assets")
    out_dir.mkdir(exist_ok=True)
    result.experiment_records.to_csv(out_dir / "fair_benchmark_records.csv", index=False)
    print(f"\nSaved per-experiment records to {out_dir / 'fair_benchmark_records.csv'}")


if __name__ == "__main__":
    main()

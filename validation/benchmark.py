"""
Automated Synthetic Validation Benchmark Suite.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .drift_injector import SyntheticDriftInjector, DriftType, InjectedDriftScenario
from detection.detector import DriftDetector
from attribution.ranker import DriftAttributionEngine, UnifiedAttributionReport

logger = logging.getLogger(__name__)


@dataclass
class SingleExperimentResult:
    scenario_id: str
    target_feature: str
    drift_type: str
    magnitude: float
    psi_detected: float
    is_alert_triggered: bool
    # Predictions
    top1_unified: str
    top3_unified: List[str]
    rank_unified: int
    top1_shap: str
    top3_shap: List[str]
    rank_shap: int
    top1_causal: str
    top3_causal: List[str]
    rank_causal: int
    # Accuracy flags
    is_top1_correct_unified: bool
    is_top3_hit_unified: bool
    is_top1_correct_shap: bool
    is_top3_hit_shap: bool
    is_top1_correct_causal: bool
    is_top3_hit_causal: bool


@dataclass
class BenchmarkEvaluationResult:
    total_experiments: int
    precision_at_1_unified: float
    recall_at_3_unified: float
    mrr_unified: float
    precision_at_1_shap: float
    recall_at_3_shap: float
    mrr_shap: float
    precision_at_1_causal: float
    recall_at_3_causal: float
    mrr_causal: float
    experiment_records: pd.DataFrame
    summary_by_feature: pd.DataFrame
    summary_by_magnitude: pd.DataFrame


class AttributionBenchmarkRunner:
    """
    Executes an extensive grid of synthetic drift injection experiments to empirically
    validate and benchmark the accuracy of root-cause attribution methods.
    """

    def __init__(
        self,
        model_wrapper,
        detector: DriftDetector,
        attributor: DriftAttributionEngine,
        injector: Optional[SyntheticDriftInjector] = None,
    ):
        self.model_wrapper = model_wrapper
        self.detector = detector
        self.attributor = attributor
        self.injector = injector or SyntheticDriftInjector(random_seed=42)

    def run_benchmark_grid(
        self,
        heldout_clean_df: pd.DataFrame,
        features_to_test: Optional[List[str]] = None,
        magnitudes: Optional[List[float]] = None,
        drift_types: Optional[List[DriftType]] = None,
    ) -> BenchmarkEvaluationResult:
        """
        Runs comprehensive test grid across features, drift types, and magnitudes.
        """
        features = features_to_test or [
            "trip_distance",
            "fare_amount",
            "speed_mph",
            "trip_duration_min",
            "extra",
            "congestion_surcharge",
            "payment_type",
            "passenger_count",
        ]
        mags = magnitudes or [0.75, 1.5, 2.5, 4.0]
        dtypes = drift_types or [DriftType.MEAN_SHIFT, DriftType.VARIANCE_SCALE, DriftType.NONLINEAR_TRANSFORM]

        results: List[SingleExperimentResult] = []
        logger.info("Starting Attribution Benchmark: %d features x %d magnitudes x %d drift types...", len(features), len(mags), len(dtypes))

        for feat in features:
            if feat not in heldout_clean_df.columns:
                continue

            for dt in dtypes:
                # Skip incompatible combinations
                if dt == DriftType.NONLINEAR_TRANSFORM and feat in ("payment_type", "passenger_count"):
                    continue

                for mag in mags:
                    scenario = self.injector.inject_drift(
                        clean_df=heldout_clean_df,
                        target_feature=feat,
                        drift_type=dt,
                        magnitude=mag,
                    )

                    # 1. Run Detection
                    preds = self.model_wrapper.predict(scenario.drifted_df)
                    uncs = self.model_wrapper.estimate_uncertainty(scenario.drifted_df)
                    drift_report = self.detector.inspect_window(
                        prod_df=scenario.drifted_df,
                        window_name=scenario.scenario_id,
                        prod_predictions=preds,
                        prod_uncertainties=uncs,
                    )

                    psi_val = drift_report.feature_metrics[feat].psi_value if feat in drift_report.feature_metrics else 0.0
                    is_alert = drift_report.overall_status in ("WARNING", "CRITICAL")

                    # 2. Run Attribution Engine
                    attr_report = self.attributor.attribute_drift(
                        prod_df=scenario.drifted_df,
                        drift_report=drift_report,
                        target_col=self.model_wrapper.target_col,
                    )

                    # Extract Unified Ranking
                    unified_ranks = [k for k, _ in sorted(attr_report.root_cause_scores.items(), key=lambda x: x[1], reverse=True)]
                    top1_u = unified_ranks[0]
                    top3_u = unified_ranks[:3]
                    rank_u = unified_ranks.index(feat) + 1 if feat in unified_ranks else len(unified_ranks)

                    # Extract SHAP Ranking
                    shap_ranks = [k for k, _ in attr_report.shap_attribution.ranked_features]
                    top1_s = shap_ranks[0]
                    top3_s = shap_ranks[:3]
                    rank_s = shap_ranks.index(feat) + 1 if feat in shap_ranks else len(shap_ranks)

                    # Extract Causal Ablation Ranking
                    causal_ranks = [k for k, _ in attr_report.causal_ablation.ranked_causal_features]
                    top1_c = causal_ranks[0]
                    top3_c = causal_ranks[:3]
                    rank_c = causal_ranks.index(feat) + 1 if feat in causal_ranks else len(causal_ranks)

                    exp_res = SingleExperimentResult(
                        scenario_id=scenario.scenario_id,
                        target_feature=feat,
                        drift_type=dt.value,
                        magnitude=mag,
                        psi_detected=psi_val,
                        is_alert_triggered=is_alert,
                        top1_unified=top1_u,
                        top3_unified=top3_u,
                        rank_unified=rank_u,
                        top1_shap=top1_s,
                        top3_shap=top3_s,
                        rank_shap=rank_s,
                        top1_causal=top1_c,
                        top3_causal=top3_c,
                        rank_causal=rank_c,
                        is_top1_correct_unified=(top1_u == feat),
                        is_top3_hit_unified=(feat in top3_u),
                        is_top1_correct_shap=(top1_s == feat),
                        is_top3_hit_shap=(feat in top3_s),
                        is_top1_correct_causal=(top1_c == feat),
                        is_top3_hit_causal=(feat in top3_c),
                    )
                    results.append(exp_res)

        records_df = pd.DataFrame([vars(r) for r in results])
        n_total = len(records_df)

        # Metrics computation
        p1_u = float(records_df["is_top1_correct_unified"].mean())
        r3_u = float(records_df["is_top3_hit_unified"].mean())
        mrr_u = float((1.0 / records_df["rank_unified"]).mean())

        p1_s = float(records_df["is_top1_correct_shap"].mean())
        r3_s = float(records_df["is_top3_hit_shap"].mean())
        mrr_s = float((1.0 / records_df["rank_shap"]).mean())

        p1_c = float(records_df["is_top1_correct_causal"].mean())
        r3_c = float(records_df["is_top3_hit_causal"].mean())
        mrr_c = float((1.0 / records_df["rank_causal"]).mean())

        summary_feat = records_df.groupby("target_feature").agg({
            "is_top1_correct_unified": "mean",
            "is_top3_hit_unified": "mean",
            "rank_unified": "mean",
            "psi_detected": "mean",
        }).reset_index().rename(columns={
            "is_top1_correct_unified": "Precision@1",
            "is_top3_hit_unified": "Recall@3",
            "rank_unified": "Avg_Rank",
            "psi_detected": "Avg_PSI",
        })

        summary_mag = records_df.groupby("magnitude").agg({
            "is_top1_correct_unified": "mean",
            "is_top1_correct_shap": "mean",
            "is_top1_correct_causal": "mean",
            "is_top3_hit_unified": "mean",
        }).reset_index().rename(columns={
            "is_top1_correct_unified": "Unified_Precision@1",
            "is_top1_correct_shap": "SHAP_Precision@1",
            "is_top1_correct_causal": "Causal_Precision@1",
            "is_top3_hit_unified": "Recall@3",
        })

        logger.info("=== BENCHMARK SUMMARY (N=%d) ===", n_total)
        logger.info("Unified Engine -> Precision@1: %.2f%%, Recall@3: %.2f%%, MRR: %.4f", p1_u * 100, r3_u * 100, mrr_u)
        logger.info("SHAP-Only      -> Precision@1: %.2f%%, Recall@3: %.2f%%, MRR: %.4f", p1_s * 100, r3_s * 100, mrr_s)
        logger.info("Causal-Only    -> Precision@1: %.2f%%, Recall@3: %.2f%%, MRR: %.4f", p1_c * 100, r3_c * 100, mrr_c)

        return BenchmarkEvaluationResult(
            total_experiments=n_total,
            precision_at_1_unified=p1_u,
            recall_at_3_unified=r3_u,
            mrr_unified=mrr_u,
            precision_at_1_shap=p1_s,
            recall_at_3_shap=r3_s,
            mrr_shap=mrr_s,
            precision_at_1_causal=p1_c,
            recall_at_3_causal=r3_c,
            mrr_causal=mrr_c,
            experiment_records=records_df,
            summary_by_feature=summary_feat,
            summary_by_magnitude=summary_mag,
        )


def generate_benchmark_visualizations(
    bench_result: BenchmarkEvaluationResult,
    save_dir: str = "./assets",
) -> List[str]:
    """Generates high-resolution publication-quality benchmark plots for README and docs."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    saved_paths = []

    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 1. Overall Method Comparison Chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_data = pd.DataFrame({
        "Method": ["Unified Engine\n(SHAP + Causal)", "Causal Ablation\n(Interventional)", "SHAP Shift\n(Observational)"],
        "Precision@1": [bench_result.precision_at_1_unified * 100, bench_result.precision_at_1_causal * 100, bench_result.precision_at_1_shap * 100],
        "Recall@3": [bench_result.recall_at_3_unified * 100, bench_result.recall_at_3_causal * 100, bench_result.recall_at_3_shap * 100],
        "MRR": [bench_result.mrr_unified, bench_result.mrr_causal, bench_result.mrr_shap],
    })

    palette = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    # Precision@1 plot
    sns.barplot(data=metrics_data, x="Method", y="Precision@1", ax=axes[0], palette=palette)
    axes[0].set_title("Top-1 Root Cause Precision (%)", weight="bold")
    axes[0].set_ylim(0, 105)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1.5), ha='center', weight='bold')

    # Recall@3 plot
    sns.barplot(data=metrics_data, x="Method", y="Recall@3", ax=axes[1], palette=palette)
    axes[1].set_title("Top-3 Root Cause Recall (%)", weight="bold")
    axes[1].set_ylim(0, 105)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1.5), ha='center', weight='bold')

    # MRR plot
    sns.barplot(data=metrics_data, x="Method", y="MRR", ax=axes[2], palette=palette)
    axes[2].set_title("Mean Reciprocal Rank (MRR)", weight="bold")
    axes[2].set_ylim(0, 1.05)
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02), ha='center', weight='bold')

    plt.tight_layout()
    chart1_path = os.path.join(save_dir, "synthetic_validation_benchmark.png")
    fig.savefig(chart1_path, dpi=200)
    plt.close(fig)
    saved_paths.append(chart1_path)

    # 2. Performance by Magnitude Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    mag_df = bench_result.summary_by_magnitude.copy()
    mag_melted = pd.melt(
        mag_df,
        id_vars=["magnitude"],
        value_vars=["Unified_Precision@1", "Causal_Precision@1", "SHAP_Precision@1"],
        var_name="Method",
        value_name="Precision",
    )
    mag_melted["Precision"] *= 100.0
    mag_melted["Method"] = mag_melted["Method"].map({
        "Unified_Precision@1": "Unified Engine (SHAP + Causal)",
        "Causal_Precision@1": "Causal Counterfactual Ablation",
        "SHAP_Precision@1": "SHAP Shift Decomposition",
    })

    sns.lineplot(data=mag_melted, x="magnitude", y="Precision", hue="Method", marker="o", linewidth=2.5, ax=ax)
    ax.set_title("Root-Cause Attribution Precision@1 vs. Injected Drift Magnitude", weight="bold")
    ax.set_xlabel("Injected Drift Magnitude (Standard Deviations / Multiplier)", weight="bold")
    ax.set_ylabel("Precision@1 (%)", weight="bold")
    ax.set_ylim(40, 105)
    plt.tight_layout()
    chart2_path = os.path.join(save_dir, "precision_vs_magnitude.png")
    fig.savefig(chart2_path, dpi=200)
    plt.close(fig)
    saved_paths.append(chart2_path)

    logger.info("Saved benchmark visualization assets to %s", save_dir)
    return saved_paths


def run_full_validation_suite(
    model_wrapper,
    detector: DriftDetector,
    attributor: DriftAttributionEngine,
    clean_val_df: pd.DataFrame,
    save_dir: str = "./assets",
) -> BenchmarkEvaluationResult:
    """Convenience entrypoint to execute full validation suite and export artifacts."""
    runner = AttributionBenchmarkRunner(model_wrapper, detector, attributor)
    result = runner.run_benchmark_grid(clean_val_df)
    generate_benchmark_visualizations(result, save_dir=save_dir)
    return result

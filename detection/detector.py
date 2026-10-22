"""
Unified Multi-Signal Drift Detection Orchestrator.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .statistical_tests import (
    calculate_psi_numerical,
    calculate_psi_categorical,
    calculate_ks_test,
    apply_benjamini_hochberg_correction,
    calculate_wasserstein_distance,
    PSIBinningResult,
)
from .uncertainty_proxy import UncertaintyDriftDetector, UncertaintyReport

logger = logging.getLogger(__name__)


@dataclass
class FeatureDriftMetric:
    feature_name: str
    feature_type: str  # "numerical" or "categorical"
    psi_value: float
    psi_status: str  # "STABLE", "WARNING", "CRITICAL"
    ks_statistic: float
    ks_pvalue: float
    ks_fdr_significant: bool
    ks_fdr_qvalue: float
    wasserstein_dist: float
    status: str  # "STABLE", "WARNING", "CRITICAL"


@dataclass
class DriftReport:
    window_name: str
    timestamp: str
    sample_size: int
    overall_status: str  # "NORMAL", "WARNING", "CRITICAL"
    summary: str
    n_drifted_features: int
    feature_metrics: Dict[str, FeatureDriftMetric]
    uncertainty_report: Optional[UncertaintyReport] = None
    supervised_metrics: Optional[Dict[str, float]] = None


class DriftDetector:
    """
    Production-grade multi-signal drift detection orchestrator.
    Combines input feature PSI, KS-testing with Benjamini-Hochberg FDR control,
    prediction drift, and epistemic uncertainty proxies.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        psi_threshold_warning: float = 0.10,
        psi_threshold_critical: float = 0.25,
        ks_alpha: float = 0.01,
        uncertainty_detector: Optional[UncertaintyDriftDetector] = None,
    ):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.psi_warn = psi_threshold_warning
        self.psi_crit = psi_threshold_critical
        self.ks_alpha = ks_alpha
        self.uncertainty_detector = uncertainty_detector or UncertaintyDriftDetector()

        self.ref_data: Optional[pd.DataFrame] = None
        self.ref_bin_edges: Dict[str, np.ndarray] = {}
        self.is_fitted = False

    def fit_reference(
        self,
        ref_df: pd.DataFrame,
        ref_predictions: Optional[np.ndarray] = None,
        ref_uncertainties: Optional[np.ndarray] = None,
    ) -> "DriftDetector":
        """Pre-computes quantile bin edges and baseline distributions on Reference Window W0."""
        self.ref_data = ref_df.copy()

        # Cache reference bin edges for all numerical features
        for feat in self.numerical_features:
            if feat in ref_df.columns:
                res = calculate_psi_numerical(ref_df[feat], ref_df[feat], n_bins=10)
                self.ref_bin_edges[feat] = res.bin_edges

        # Fit uncertainty detector if predictions are provided
        if ref_predictions is not None and ref_uncertainties is not None:
            all_feats = self.numerical_features + self.categorical_features
            ref_feat_arr = ref_df[all_feats].to_numpy() if all(c in ref_df.columns for c in all_feats) else None
            self.uncertainty_detector.fit_reference(ref_predictions, ref_uncertainties, ref_feat_arr)

        self.is_fitted = True
        logger.info("DriftDetector fitted on reference data (%d samples, %d features).", len(ref_df), len(self.numerical_features) + len(self.categorical_features))
        return self

    def inspect_window(
        self,
        prod_df: pd.DataFrame,
        window_name: str = "production_batch",
        prod_predictions: Optional[np.ndarray] = None,
        prod_uncertainties: Optional[np.ndarray] = None,
        supervised_metrics: Optional[Dict[str, float]] = None,
    ) -> DriftReport:
        """
        Executes full statistical detection suite across input features,
        model predictions, and uncertainty proxies.
        """
        if not self.is_fitted or self.ref_data is None:
            raise RuntimeError("DriftDetector must be fitted on reference data first.")

        feature_metrics: Dict[str, FeatureDriftMetric] = {}
        raw_pvalues: Dict[str, float] = {}

        # 1. Evaluate Numerical Features (PSI + KS test + Wasserstein)
        for feat in self.numerical_features:
            if feat not in prod_df.columns or feat not in self.ref_data.columns:
                continue

            bin_edges = self.ref_bin_edges.get(feat)
            psi_res = calculate_psi_numerical(self.ref_data[feat], prod_df[feat], bin_edges=bin_edges)
            ks_stat, ks_pval, ks_status = calculate_ks_test(self.ref_data[feat], prod_df[feat])
            raw_pvalues[feat] = ks_pval
            wass_dist = calculate_wasserstein_distance(self.ref_data[feat], prod_df[feat])

            # Determine composite feature status
            if psi_res.status == "CRITICAL" or ks_status == "CRITICAL":
                feat_status = "CRITICAL"
            elif psi_res.status == "WARNING" or ks_status == "WARNING":
                feat_status = "WARNING"
            else:
                feat_status = "STABLE"

            feature_metrics[feat] = FeatureDriftMetric(
                feature_name=feat,
                feature_type="numerical",
                psi_value=psi_res.psi_value,
                psi_status=psi_res.status,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                ks_fdr_significant=False,  # updated via BH step
                ks_fdr_qvalue=ks_pval,
                wasserstein_dist=wass_dist,
                status=feat_status,
            )

        # 2. Apply Benjamini-Hochberg FDR correction to KS tests
        fdr_results = apply_benjamini_hochberg_correction(raw_pvalues, alpha=self.ks_alpha)
        for feat, (is_sig, q_val) in fdr_results.items():
            if feat in feature_metrics:
                feature_metrics[feat].ks_fdr_significant = is_sig
                feature_metrics[feat].ks_fdr_qvalue = q_val

        # 3. Evaluate Categorical Features (Categorical PSI)
        for feat in self.categorical_features:
            if feat not in prod_df.columns or feat not in self.ref_data.columns:
                continue

            psi_res = calculate_psi_categorical(self.ref_data[feat], prod_df[feat])
            feature_metrics[feat] = FeatureDriftMetric(
                feature_name=feat,
                feature_type="categorical",
                psi_value=psi_res.psi_value,
                psi_status=psi_res.status,
                ks_statistic=0.0,
                ks_pvalue=1.0,
                ks_fdr_significant=False,
                ks_fdr_qvalue=1.0,
                wasserstein_dist=0.0,
                status=psi_res.status,
            )

        # 4. Evaluate Unsupervised / Uncertainty Proxy
        unc_report = None
        if prod_predictions is not None and prod_uncertainties is not None and self.uncertainty_detector.is_fitted:
            all_feats = self.numerical_features + self.categorical_features
            prod_feat_arr = prod_df[all_feats].to_numpy() if all(c in prod_df.columns for c in all_feats) else None
            unc_report = self.uncertainty_detector.evaluate(prod_predictions, prod_uncertainties, prod_feat_arr)

        # 5. Aggregate Global Alert Status
        critical_count = sum(1 for m in feature_metrics.values() if m.status == "CRITICAL")
        warning_count = sum(1 for m in feature_metrics.values() if m.status == "WARNING")
        drifted_total = critical_count + warning_count

        if critical_count > 0 or (unc_report and unc_report.alert_status == "CRITICAL"):
            overall_status = "CRITICAL"
            summary = f"CRITICAL ALERT: {critical_count} features exhibiting severe drift (PSI >= 0.25)."
            if unc_report and unc_report.alert_status == "CRITICAL":
                summary += f" Model uncertainty/prediction drift critical ({unc_report.justification})."
        elif warning_count > 0 or (unc_report and unc_report.alert_status == "WARNING"):
            overall_status = "WARNING"
            summary = f"WARNING: {warning_count} features showing moderate covariate shift (0.10 <= PSI < 0.25)."
        else:
            overall_status = "NORMAL"
            summary = "NOMINAL: All monitored features and uncertainty signals within reference thresholds."

        return DriftReport(
            window_name=window_name,
            timestamp=pd.Timestamp.now().isoformat(),
            sample_size=len(prod_df),
            overall_status=overall_status,
            summary=summary,
            n_drifted_features=drifted_total,
            feature_metrics=feature_metrics,
            uncertainty_report=unc_report,
            supervised_metrics=supervised_metrics,
        )

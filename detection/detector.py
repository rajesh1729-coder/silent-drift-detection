"""
Unified Multi-Signal Drift Detection Orchestrator.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .statistical_tests import (
    calculate_psi_numerical,
    calculate_psi_categorical,
    calculate_ks_test,
    apply_benjamini_hochberg_correction,
    calculate_wasserstein_distance,
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
    ks_status: str = "STABLE"


def combine_feature_status(psi_status: str, ks_status: str, ks_fdr_significant: bool) -> str:
    """
    Decide one feature's final status.
    PSI always counts. The KS verdict only counts if the Benjamini-Hochberg
    procedure also found that feature's KS test significant.
    """
    effective_ks = ks_status if ks_fdr_significant else "STABLE"
    if "CRITICAL" in (psi_status, effective_ks):
        return "CRITICAL"
    if "WARNING" in (psi_status, effective_ks):
        return "WARNING"
    return "STABLE"


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
        self.uncertainty_detector = uncertainty_detector or UncertaintyDriftDetector(
            psi_threshold_warning=psi_threshold_warning,
            psi_threshold_critical=psi_threshold_critical,
        )

        self.ref_data: Optional[pd.DataFrame] = None
        self.ref_bin_edges: Dict[str, Optional[np.ndarray]] = {}
        self.is_fitted = False

    def fit_reference(
        self,
        ref_df: pd.DataFrame,
        ref_predictions: Optional[np.ndarray] = None,
        ref_uncertainties: Optional[np.ndarray] = None,
    ) -> "DriftDetector":
        """Pre-computes quantile bin edges and baseline distributions on Reference Window W0."""
        self.ref_data = ref_df.copy()

        for feat in self.numerical_features:
            if feat in ref_df.columns:
                clean_col = ref_df[feat].dropna()
                if len(clean_col.unique()) > 10:
                    res = calculate_psi_numerical(
                        ref_df[feat],
                        ref_df[feat],
                        n_bins=10,
                        warn_threshold=self.psi_warn,
                        crit_threshold=self.psi_crit,
                    )
                    self.ref_bin_edges[feat] = res.bin_edges
                else:
                    self.ref_bin_edges[feat] = None

        if ref_predictions is not None and ref_uncertainties is not None:
            all_feats = self.numerical_features + self.categorical_features
            ref_feat_arr = ref_df[all_feats].to_numpy() if all(c in ref_df.columns for c in all_feats) else None
            self.uncertainty_detector.fit_reference(ref_predictions, ref_uncertainties, ref_feat_arr)

        self.is_fitted = True
        logger.info(
            "DriftDetector fitted on reference data (%d samples, %d features).",
            len(ref_df),
            len(self.numerical_features) + len(self.categorical_features),
        )
        return self

    def inspect_window(
        self,
        prod_df: pd.DataFrame,
        window_name: str = "production_batch",
        prod_predictions: Optional[np.ndarray] = None,
        prod_uncertainties: Optional[np.ndarray] = None,
        supervised_metrics: Optional[Dict[str, float]] = None,
    ) -> DriftReport:
        """Executes full statistical detection suite across input features, predictions, and uncertainty."""
        if not self.is_fitted or self.ref_data is None:
            raise RuntimeError("DriftDetector must be fitted on reference data first.")

        feature_metrics: Dict[str, FeatureDriftMetric] = {}
        raw_pvalues: Dict[str, float] = {}

        # 1. Evaluate Numerical Features
        for feat in self.numerical_features:
            if feat not in prod_df.columns or feat not in self.ref_data.columns:
                continue

            bin_edges = self.ref_bin_edges.get(feat)
            psi_res = calculate_psi_numerical(
                self.ref_data[feat],
                prod_df[feat],
                bin_edges=bin_edges,
                warn_threshold=self.psi_warn,
                crit_threshold=self.psi_crit,
            )
            ks_stat, ks_pval, ks_status = calculate_ks_test(self.ref_data[feat], prod_df[feat])
            raw_pvalues[feat] = ks_pval
            wass_dist = calculate_wasserstein_distance(self.ref_data[feat], prod_df[feat])

            feature_metrics[feat] = FeatureDriftMetric(
                feature_name=feat,
                feature_type="numerical",
                psi_value=psi_res.psi_value,
                psi_status=psi_res.status,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                ks_fdr_significant=False,
                ks_fdr_qvalue=ks_pval,
                wasserstein_dist=wass_dist,
                status=psi_res.status,
                ks_status=ks_status,
            )

        # 2. Apply Benjamini-Hochberg FDR correction to KS tests
        fdr_results = apply_benjamini_hochberg_correction(raw_pvalues, alpha=self.ks_alpha)
        for feat, (is_sig, q_val) in fdr_results.items():
            if feat in feature_metrics:
                m = feature_metrics[feat]
                m.ks_fdr_significant = is_sig
                m.ks_fdr_qvalue = q_val
                m.status = combine_feature_status(m.psi_status, m.ks_status, is_sig)

        # 3. Evaluate Categorical Features
        for feat in self.categorical_features:
            if feat not in prod_df.columns or feat not in self.ref_data.columns:
                continue

            psi_res = calculate_psi_categorical(
                self.ref_data[feat],
                prod_df[feat],
                warn_threshold=self.psi_warn,
                crit_threshold=self.psi_crit,
            )
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
        critical_feats = [f for f, m in feature_metrics.items() if m.status == "CRITICAL"]
        warning_feats = [f for f, m in feature_metrics.items() if m.status == "WARNING"]
        critical_count = len(critical_feats)
        warning_count = len(warning_feats)
        drifted_total = critical_count + warning_count
        unc_status = unc_report.alert_status if unc_report else "STABLE"

        def _names(feats: List[str]) -> str:
            shown = ", ".join(feats[:5])
            return shown + (f", +{len(feats) - 5} more" if len(feats) > 5 else "")

        if critical_count > 0 or unc_status == "CRITICAL":
            overall_status = "CRITICAL"
            parts = []
            if critical_count:
                parts.append(f"{critical_count} critical feature(s): {_names(critical_feats)}")
            if warning_count:
                parts.append(f"{warning_count} warning feature(s)")
            summary = "CRITICAL ALERT: " + ("; ".join(parts) if parts else "model signals only") + "."
            if unc_status == "CRITICAL" and unc_report is not None:
                summary += f" Model uncertainty/prediction drift critical ({unc_report.justification})."
        elif warning_count > 0 or unc_status == "WARNING":
            overall_status = "WARNING"
            summary = "WARNING:"
            if warning_count:
                summary += f" {warning_count} feature(s) with moderate drift: {_names(warning_feats)}."
            if unc_status == "WARNING" and unc_report is not None:
                summary += f" Model uncertainty/prediction drift warning ({unc_report.justification})."
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
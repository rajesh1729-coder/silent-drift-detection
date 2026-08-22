"""
Unified Root-Cause Attribution Engine.
Combines SHAP marginal distribution shift with Causal Counterfactual Ablation.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .shap_attribution import SHAPDriftAttributor, SHAPAttributionResult
from .causal_ablation import CausalAblationAttributor, CausalAblationResult
from detection.detector import DriftReport

logger = logging.getLogger(__name__)


@dataclass
class FeatureAttributionSummary:
    feature_name: str
    psi_value: float
    ks_pvalue: float
    shap_delta_mean: float
    shap_relative_share: float
    shap_wasserstein: float
    causal_pred_recovery: float
    causal_error_recovery: float
    root_cause_score: float
    rank: int
    is_top_root_cause: bool
    diagnostic_narrative: str


@dataclass
class UnifiedAttributionReport:
    window_name: str
    overall_alert_status: str
    top_root_cause_feature: str
    top_3_root_causes: List[str]
    root_cause_scores: Dict[str, float]
    feature_summaries: Dict[str, FeatureAttributionSummary]
    shap_attribution: SHAPAttributionResult
    causal_ablation: CausalAblationResult
    executive_summary: str


class DriftAttributionEngine:
    """
    Coordinates SHAP-based explanatory shift decomposition and Causal Counterfactual
    ablation to provide definitive, ranked root-cause attributions for detected drift alerts.
    """

    def __init__(
        self,
        model_wrapper,
        shap_weight: float = 0.40,
        causal_weight: float = 0.60,
        max_samples: int = 2000,
    ):
        self.model_wrapper = model_wrapper
        self.shap_weight = shap_weight
        self.causal_weight = causal_weight
        self.shap_attributor = SHAPDriftAttributor(model_wrapper, max_samples=max_samples)
        self.causal_attributor = CausalAblationAttributor(model_wrapper, max_samples=max_samples)
        self.is_fitted = False

    def fit_reference(self, ref_df: pd.DataFrame) -> "DriftAttributionEngine":
        """Fits both SHAP and Causal ablation baselines on the Reference Window."""
        self.shap_attributor.fit_reference(ref_df)
        self.causal_attributor.fit_reference(ref_df)
        self.is_fitted = True
        logger.info("DriftAttributionEngine successfully initialized with reference data.")
        return self

    def attribute_drift(
        self,
        prod_df: pd.DataFrame,
        drift_report: Optional[DriftReport] = None,
        target_col: Optional[str] = None,
    ) -> UnifiedAttributionReport:
        """
        Executes unified attribution pipeline on the drifted production window.
        """
        if not self.is_fitted:
            raise RuntimeError("DriftAttributionEngine must be fitted on reference data first.")

        window_name = drift_report.window_name if drift_report else "production_batch"
        alert_status = drift_report.overall_status if drift_report else "WARNING"

        # Determine suspect candidate features
        candidate_features = None
        if drift_report is not None and drift_report.n_drifted_features > 0:
            candidate_features = [
                f for f, m in drift_report.feature_metrics.items()
                if m.status in ("WARNING", "CRITICAL")
            ]
        
        # Fallback to all features if none flagged or candidate list empty
        if not candidate_features:
            candidate_features = self.model_wrapper.features

        # 1. Compute SHAP Distribution Shift
        shap_res = self.shap_attributor.attribute(prod_df)

        # 2. Compute Causal Counterfactual Ablation
        causal_res = self.causal_attributor.attribute(
            prod_df,
            candidate_features=candidate_features,
            target_col=target_col,
        )

        # 3. Blend Scores into Unified Root Cause Ranking
        features = self.model_wrapper.features
        combined_scores: Dict[str, float] = {}

        for feat in features:
            s_score = shap_res.relative_attribution_share.get(feat, 0.0)
            c_score = causal_res.prediction_recovery_scores.get(feat, 0.0)
            
            # If error recovery is available from labels, blend with error recovery
            if target_col and feat in causal_res.error_recovery_scores:
                c_err = causal_res.error_recovery_scores[feat]
                c_effective = 0.5 * c_score + 0.5 * c_err
            else:
                c_effective = c_score

            score = (self.shap_weight * s_score) + (self.causal_weight * c_effective)
            combined_scores[feat] = float(score)

        # Sort features by combined score
        sorted_ranks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_feature = sorted_ranks[0][0]
        top_3_features = [x[0] for x in sorted_ranks[:3]]

        # Build feature summaries
        summaries: Dict[str, FeatureAttributionSummary] = {}
        for rank_idx, (feat, score) in enumerate(sorted_ranks, start=1):
            psi_val = 0.0
            ks_pval = 1.0
            if drift_report and feat in drift_report.feature_metrics:
                psi_val = drift_report.feature_metrics[feat].psi_value
                ks_pval = drift_report.feature_metrics[feat].ks_pvalue

            shap_delta = shap_res.delta_mean_shaps.get(feat, 0.0)
            shap_share = shap_res.relative_attribution_share.get(feat, 0.0)
            shap_wass = shap_res.shap_wasserstein_dists.get(feat, 0.0)
            c_pred = causal_res.prediction_recovery_scores.get(feat, 0.0)
            c_err = causal_res.error_recovery_scores.get(feat, 0.0)

            # Generate narrative
            if rank_idx == 1:
                narrative = (
                    f"PRIMARY ROOT CAUSE: Feature '{feat}' accounts for {shap_share*100:.1f}% of SHAP shift "
                    f"and closes {c_pred*100:.1f}% of prediction drift under counterfactual intervention."
                )
            elif rank_idx <= 3 and score > 0.15:
                narrative = (
                    f"SECONDARY CONTRIBUTOR: Feature '{feat}' accounts for {shap_share*100:.1f}% of SHAP shift "
                    f"and closes {c_pred*100:.1f}% of drift under counterfactual ablation."
                )
            else:
                narrative = f"Minor contributor (Attribution score = {score:.3f})."

            summaries[feat] = FeatureAttributionSummary(
                feature_name=feat,
                psi_value=psi_val,
                ks_pvalue=ks_pval,
                shap_delta_mean=shap_delta,
                shap_relative_share=shap_share,
                shap_wasserstein=shap_wass,
                causal_pred_recovery=c_pred,
                causal_error_recovery=c_err,
                root_cause_score=score,
                rank=rank_idx,
                is_top_root_cause=(rank_idx == 1),
                diagnostic_narrative=narrative,
            )

        exec_summary = (
            f"Window '{window_name}' analyzed with status {alert_status}. "
            f"Root cause attribution identified '{top_feature}' as the leading causal driver "
            f"(Score: {combined_scores[top_feature]:.3f}, Top 3: {', '.join(top_3_features)})."
        )

        return UnifiedAttributionReport(
            window_name=window_name,
            overall_alert_status=alert_status,
            top_root_cause_feature=top_feature,
            top_3_root_causes=top_3_features,
            root_cause_scores=combined_scores,
            feature_summaries=summaries,
            shap_attribution=shap_res,
            causal_ablation=causal_res,
            executive_summary=exec_summary,
        )

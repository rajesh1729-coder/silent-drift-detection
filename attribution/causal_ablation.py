"""
Causal Counterfactual Ablation Engine.

Causal Reasoning & Interventional Mechanics:
--------------------------------------------
Why Observational Statistics Fail to Identify True Root Causes:
1. Spurious Correlation: If a feature Z is correlated with the true drifted feature X (e.g.
   fare_amount and trip_distance), observational tests (PSI, KS) and model explanations (SHAP)
   will flag BOTH features as drifted.
2. Downstream Covariate Shift: A model might be highly sensitive to a non-causal downstream feature.
   Flagging it leads engineers to fix the wrong operational pipeline.

The Causal Counterfactual Intervention:
To establish causality, we intervene on the system using Pearl's do-calculus framework:
    P( Y | do(X_j = Reference) )

We generate counterfactual production batches where candidate feature X_j is intervened upon
and mapped back to its reference marginal distribution P_ref(X_j), while keeping all other
production features untouched.

Two Counterfactual Mechanisms Implemented:
1. Quantile Mapping (1D Optimal Transport):
   Maps production values x_i to the exact reference quantile:
       x_i^{cf} = F_ref^{-1}( F_prod(x_i) )
   This preserves rank structure while restoring exact baseline distribution.

2. Empirical Reference Resampling:
   Replaces X_j with i.i.d. draws from the reference population.

Causal Impact Metric:
Measures the percentage of the prediction shift or loss degradation that is recovered
when feature X_j is intervened on.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


@dataclass
class CausalAblationResult:
    feature_names: List[str]
    candidate_features: List[str]
    prediction_recovery_scores: Dict[str, float]  # share of prediction shift closed by ablating X_j
    error_recovery_scores: Dict[str, float]  # share of RMSE degradation closed (if labels available)
    ablated_pred_means: Dict[str, float]
    ablated_rmses: Dict[str, float]
    ranked_causal_features: List[Tuple[str, float]]  # sorted by causal recovery impact


class CausalAblationAttributor:
    """
    Evaluates causal root-cause contributions by performing counterfactual interventions
    on candidate features and measuring prediction/loss gap closure.
    """

    def __init__(self, model_wrapper, max_samples: int = 2500, intervention_method: str = "quantile_matching"):
        self.model_wrapper = model_wrapper
        self.max_samples = max_samples
        self.method = intervention_method  # "quantile_matching" or "resampling"
        self.ref_data: Optional[pd.DataFrame] = None
        self.is_fitted = False

    def fit_reference(self, ref_df: pd.DataFrame) -> "CausalAblationAttributor":
        """Stores reference baseline features for counterfactual synthesis."""
        features = self.model_wrapper.features
        self.ref_data = ref_df[features].copy()
        if len(self.ref_data) > self.max_samples:
            self.ref_data = self.ref_data.sample(n=self.max_samples, random_state=42).reset_index(drop=True)
        self.is_fitted = True
        return self

    def _intervene_feature(
        self,
        prod_df: pd.DataFrame,
        feature_name: str,
    ) -> pd.DataFrame:
        """
        Creates counterfactual dataset by holding feature_name at its reference distribution.
        """
        cf_df = prod_df.copy()
        ref_vals = self.ref_data[feature_name].to_numpy()
        prod_vals = prod_df[feature_name].to_numpy()

        if self.method == "quantile_matching" and np.issubdtype(ref_vals.dtype, np.number) and len(np.unique(ref_vals)) > 10:
            # 1D Optimal Transport Quantile Mapping:
            # 1. Compute empirical percentiles of production samples in [0, 100]
            ranks = pd.Series(prod_vals).rank(pct=True, method="average").to_numpy() * 100.0
            ranks = np.clip(ranks, 0.0, 100.0)
            # 2. Map ranks to reference percentiles
            cf_vals = np.percentile(ref_vals, ranks)
            cf_df[feature_name] = cf_vals
        else:
            # Empirical Resampling / Categorical intervention
            idx = np.random.choice(len(ref_vals), size=len(prod_df), replace=True)
            cf_df[feature_name] = ref_vals[idx]

        return cf_df

    def attribute(
        self,
        prod_df: pd.DataFrame,
        candidate_features: Optional[List[str]] = None,
        target_col: Optional[str] = None,
    ) -> CausalAblationResult:
        """
        Runs causal ablation across all candidate features and computes recovery scores.
        """
        if not self.is_fitted or self.ref_data is None:
            raise RuntimeError("CausalAblationAttributor must be fitted on reference data first.")

        features = self.model_wrapper.features
        candidates = candidate_features or features

        # Subsample production batch for computational efficiency
        X_prod = prod_df.copy()
        if len(X_prod) > self.max_samples:
            X_prod = X_prod.sample(n=self.max_samples, random_state=42).reset_index(drop=True)

        # Baseline predictions & metrics
        ref_preds = self.model_wrapper.predict(self.ref_data)
        ref_pred_mean = float(np.mean(ref_preds))

        prod_preds = self.model_wrapper.predict(X_prod)
        prod_pred_mean = float(np.mean(prod_preds))
        total_pred_shift = abs(prod_pred_mean - ref_pred_mean)

        has_labels = target_col is not None and target_col in X_prod.columns
        ref_rmse = 0.0
        prod_rmse = 0.0
        if has_labels:
            y_true = X_prod[target_col].to_numpy()
            prod_rmse = float(np.sqrt(mean_squared_error(y_true, prod_preds)))
            ref_rmse = self.model_wrapper.reference_metrics.get("rmse", prod_rmse * 0.8)

        pred_recovery: Dict[str, float] = {}
        error_recovery: Dict[str, float] = {}
        ablated_means: Dict[str, float] = {}
        ablated_rmses: Dict[str, float] = {}

        raw_pred_deltas: Dict[str, float] = {}

        for feat in candidates:
            if feat not in features:
                continue

            # Counterfactual intervention
            cf_batch = self._intervene_feature(X_prod, feat)
            cf_preds = self.model_wrapper.predict(cf_batch)
            cf_mean = float(np.mean(cf_preds))
            ablated_means[feat] = cf_mean

            # Prediction shift closed when intervening on feat:
            # Shift recovered = |prod_mean - cf_mean|
            delta_pred = abs(prod_pred_mean - cf_mean)
            raw_pred_deltas[feat] = delta_pred

            if has_labels:
                cf_rmse = float(np.sqrt(mean_squared_error(y_true, cf_preds)))
                ablated_rmses[feat] = cf_rmse
                # Error recovery: how much RMSE degradation is removed
                gap_total = max(1e-4, prod_rmse - ref_rmse)
                gap_closed = max(0.0, prod_rmse - cf_rmse)
                error_recovery[feat] = float(gap_closed / gap_total)
            else:
                error_recovery[feat] = 0.0

        # Normalize prediction recovery scores to relative shares
        sum_deltas = sum(raw_pred_deltas.values()) + 1e-6
        for feat in candidates:
            pred_recovery[feat] = float(raw_pred_deltas[feat] / sum_deltas)

        # Rank features primarily by causal prediction recovery or error recovery
        if has_labels and any(v > 0.05 for v in error_recovery.values()):
            ranked = sorted(error_recovery.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked = sorted(pred_recovery.items(), key=lambda x: x[1], reverse=True)

        return CausalAblationResult(
            feature_names=features,
            candidate_features=candidates,
            prediction_recovery_scores=pred_recovery,
            error_recovery_scores=error_recovery,
            ablated_pred_means=ablated_means,
            ablated_rmses=ablated_rmses,
            ranked_causal_features=ranked,
        )

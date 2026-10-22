"""
Causal Counterfactual Ablation Engine.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


@dataclass
class CausalAblationResult:
    feature_names: List[str]
    candidate_features: List[str]
    prediction_recovery_scores: Dict[str, float]
    error_recovery_scores: Dict[str, float]
    ablated_pred_means: Dict[str, float]
    ablated_rmses: Dict[str, float]
    ranked_causal_features: List[Tuple[str, float]]


class CausalAblationAttributor:
    """
    Evaluates causal root-cause contributions by performing counterfactual interventions
    on candidate features and measuring prediction/loss gap closure.
    """

    def __init__(
        self,
        model_wrapper,
        max_samples: int = 2500,
        intervention_method: str = "quantile_matching",
        random_state: int = 42,
    ):
        self.model_wrapper = model_wrapper
        self.max_samples = max_samples
        self.method = intervention_method
        self.random_state = random_state
        self.ref_data: Optional[pd.DataFrame] = None
        self.is_fitted = False

    def fit_reference(self, ref_df: pd.DataFrame) -> "CausalAblationAttributor":
        """Stores reference baseline features for counterfactual synthesis."""
        features = self.model_wrapper.features
        self.ref_data = ref_df[features].copy()
        if len(self.ref_data) > self.max_samples:
            self.ref_data = self.ref_data.sample(n=self.max_samples, random_state=self.random_state).reset_index(drop=True)
        self.is_fitted = True
        return self

    def _intervene_feature(
        self,
        prod_df: pd.DataFrame,
        feature_name: str,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Creates counterfactual dataset by holding feature_name at its reference distribution.
        """
        cf_df = prod_df.copy()
        ref_vals = self.ref_data[feature_name].to_numpy()
        prod_vals = prod_df[feature_name].to_numpy()

        if (
            self.method == "quantile_matching"
            and np.issubdtype(ref_vals.dtype, np.number)
            and len(np.unique(ref_vals)) > 10
        ):
            # 1D Optimal Transport Quantile Mapping
            ranks = pd.Series(prod_vals).rank(pct=True, method="average").to_numpy() * 100.0
            ranks = np.clip(ranks, 0.0, 100.0)
            cf_vals = np.percentile(ref_vals, ranks)
            cf_df[feature_name] = cf_vals
        else:
            # Deterministic Empirical Resampling for discrete/categorical features
            idx = rng.choice(len(ref_vals), size=len(prod_df), replace=True)
            cf_df[feature_name] = ref_vals[idx]

        return cf_df

    def attribute(
        self,
        prod_df: pd.DataFrame,
        candidate_features: Optional[List[str]] = None,
        target_col: Optional[str] = None,
    ) -> CausalAblationResult:
        """
        Runs causal ablation across candidate features and computes recovery scores.
        Guarantees complete dictionaries across all model features.
        """
        if not self.is_fitted or self.ref_data is None:
            raise RuntimeError("CausalAblationAttributor must be fitted on reference data first.")

        features = self.model_wrapper.features
        candidates = [f for f in (candidate_features or features) if f in features]
        if not candidates:
            candidates = list(features)

        X_prod = prod_df.copy()
        if len(X_prod) > self.max_samples:
            X_prod = X_prod.sample(n=self.max_samples, random_state=self.random_state).reset_index(drop=True)

        prod_preds = self.model_wrapper.predict(X_prod)
        prod_pred_mean = float(np.mean(prod_preds))

        has_labels = target_col is not None and target_col in X_prod.columns
        ref_rmse = 0.0
        prod_rmse = 0.0
        if has_labels:
            y_true = X_prod[target_col].to_numpy()
            prod_rmse = float(np.sqrt(mean_squared_error(y_true, prod_preds)))
            ref_rmse = self.model_wrapper.reference_metrics.get("rmse", prod_rmse * 0.8)

        # Initialize all model features so downstream consumers never hit KeyError
        pred_recovery: Dict[str, float] = {f: 0.0 for f in features}
        error_recovery: Dict[str, float] = {f: 0.0 for f in features}
        ablated_means: Dict[str, float] = {f: prod_pred_mean for f in features}
        ablated_rmses: Dict[str, float] = {f: prod_rmse for f in features}
        raw_pred_deltas: Dict[str, float] = {f: 0.0 for f in features}

        rng = np.random.default_rng(self.random_state)

        for feat in candidates:
            cf_batch = self._intervene_feature(X_prod, feat, rng)
            cf_preds = self.model_wrapper.predict(cf_batch)
            cf_mean = float(np.mean(cf_preds))
            ablated_means[feat] = cf_mean

            # Distribution & expectation shift recovered when intervening on feat
            mean_shift_closed = abs(prod_pred_mean - cf_mean)
            pointwise_shift_closed = float(np.mean(np.abs(prod_preds - cf_preds)))
            # Blend mean shift closure with pointwise counterfactual effect for symmetric/variance shifts
            raw_pred_deltas[feat] = 0.7 * mean_shift_closed + 0.3 * pointwise_shift_closed

            if has_labels:
                cf_rmse = float(np.sqrt(mean_squared_error(y_true, cf_preds)))
                ablated_rmses[feat] = cf_rmse
                gap_total = max(1e-4, prod_rmse - ref_rmse)
                gap_closed = max(0.0, prod_rmse - cf_rmse)
                error_recovery[feat] = float(gap_closed / gap_total)

        sum_deltas = sum(raw_pred_deltas.values()) + 1e-8
        for feat in features:
            pred_recovery[feat] = float(raw_pred_deltas[feat] / sum_deltas)

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
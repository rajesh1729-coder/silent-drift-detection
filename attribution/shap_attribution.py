"""
SHAP Distribution Shift Decomposition.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap

from detection.statistical_tests import calculate_wasserstein_distance, calculate_psi_numerical

logger = logging.getLogger(__name__)


@dataclass
class SHAPAttributionResult:
    feature_names: List[str]
    delta_expected_pred: float
    base_value: float
    ref_mean_shaps: Dict[str, float]
    prod_mean_shaps: Dict[str, float]
    delta_mean_shaps: Dict[str, float]
    relative_attribution_share: Dict[str, float]
    shap_wasserstein_dists: Dict[str, float]
    shap_psi_values: Dict[str, float]
    ranked_features: List[Tuple[str, float]]


class SHAPDriftAttributor:
    """
    Computes SHAP value distribution shifts between reference and production windows.
    """

    def __init__(self, model_wrapper, max_samples: int = 1500, random_state: int = 42):
        self.model_wrapper = model_wrapper
        self.max_samples = max_samples
        self.random_state = random_state
        self.explainer: Optional[shap.TreeExplainer] = None
        self.ref_shap_values: Optional[np.ndarray] = None
        self.ref_features_df: Optional[pd.DataFrame] = None
        self.is_fitted = False

    def fit_reference(self, ref_df: pd.DataFrame) -> "SHAPDriftAttributor":
        """Pre-computes reference SHAP distributions for baseline comparison."""
        features = self.model_wrapper.features
        X_ref = ref_df[features].copy()

        if len(X_ref) > self.max_samples:
            X_ref = X_ref.sample(n=self.max_samples, random_state=self.random_state).reset_index(drop=True)

        self.ref_features_df = X_ref
        logger.info("Initializing TreeExplainer on %d reference samples...", len(X_ref))
        self.explainer = shap.TreeExplainer(self.model_wrapper.model)
        self.ref_shap_values = np.asarray(self.explainer.shap_values(X_ref), dtype=float)
        self.is_fitted = True
        return self

    def attribute(self, prod_df: pd.DataFrame) -> SHAPAttributionResult:
        """Decomposes the prediction shift into per-feature SHAP distribution shifts."""
        if not self.is_fitted or self.explainer is None or self.ref_shap_values is None:
            raise RuntimeError("SHAPDriftAttributor must be fitted on reference data first.")

        features = self.model_wrapper.features
        X_prod = prod_df[features].copy()

        if len(X_prod) > self.max_samples:
            X_prod = X_prod.sample(n=self.max_samples, random_state=self.random_state).reset_index(drop=True)

        prod_shap_values = np.asarray(self.explainer.shap_values(X_prod), dtype=float)

        exp_val = np.asarray(self.explainer.expected_value).ravel()
        base_val = float(exp_val[0])

        ref_pred_mean = float(np.mean(self.model_wrapper.predict(self.ref_features_df)))
        prod_pred_mean = float(np.mean(self.model_wrapper.predict(X_prod)))
        delta_expected_pred = prod_pred_mean - ref_pred_mean

        ref_mean_shaps: Dict[str, float] = {}
        prod_mean_shaps: Dict[str, float] = {}
        delta_mean_shaps: Dict[str, float] = {}
        shap_wass: Dict[str, float] = {}
        shap_psi: Dict[str, float] = {}

        for idx, feat in enumerate(features):
            ref_phi = self.ref_shap_values[:, idx]
            prod_phi = prod_shap_values[:, idx]

            r_m = float(np.mean(ref_phi))
            p_m = float(np.mean(prod_phi))
            d_m = p_m - r_m

            ref_mean_shaps[feat] = r_m
            prod_mean_shaps[feat] = p_m
            delta_mean_shaps[feat] = d_m

            shap_wass[feat] = calculate_wasserstein_distance(ref_phi, prod_phi)
            psi_res = calculate_psi_numerical(ref_phi, prod_phi, n_bins=10)
            shap_psi[feat] = psi_res.psi_value

        # Combine expectation shift with distributional 1-Wasserstein energy (both in target $ units)
        abs_deltas = {f: abs(delta_mean_shaps[f]) + 0.5 * shap_wass[f] for f in features}
        total_energy = sum(abs_deltas.values()) + 1e-8
        relative_share = {f: float(val / total_energy) for f, val in abs_deltas.items()}

        ranked = sorted(relative_share.items(), key=lambda x: x[1], reverse=True)

        return SHAPAttributionResult(
            feature_names=features,
            delta_expected_pred=delta_expected_pred,
            base_value=base_val,
            ref_mean_shaps=ref_mean_shaps,
            prod_mean_shaps=prod_mean_shaps,
            delta_mean_shaps=delta_mean_shaps,
            relative_attribution_share=relative_share,
            shap_wasserstein_dists=shap_wass,
            shap_psi_values=shap_psi,
            ranked_features=ranked,
        )
r"""
Uncertainty-Based and Unsupervised Silent Drift Detectors.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .statistical_tests import (
    calculate_psi_numerical,
    calculate_ks_test,
    calculate_wasserstein_distance,
)


@dataclass
class UncertaintyReport:
    pred_psi: float
    pred_ks_stat: float
    pred_ks_pval: float
    pred_wasserstein: float
    uncertainty_psi: float
    uncertainty_mean_ratio: float
    manifold_distance_drift: float
    alert_status: str  # "STABLE", "WARNING", "CRITICAL"
    justification: str


class UncertaintyDriftDetector:
    """
    Monitors model prediction distribution, epistemic uncertainty, and standardized
    k-NN manifold distance to detect silent drift without ground truth labels.
    """

    def __init__(
        self,
        psi_threshold_warning: float = 0.10,
        psi_threshold_critical: float = 0.25,
        uncertainty_ratio_warning: float = 1.35,
        uncertainty_ratio_critical: float = 1.75,
        manifold_ratio_warning: float = 1.50,
        manifold_ratio_critical: float = 2.25,
        manifold_k_neighbors: int = 10,
        random_seed: int = 42,
    ):
        self.psi_warn = psi_threshold_warning
        self.psi_crit = psi_threshold_critical
        self.unc_warn = uncertainty_ratio_warning
        self.unc_crit = uncertainty_ratio_critical
        self.manifold_warn = manifold_ratio_warning
        self.manifold_crit = manifold_ratio_critical
        self.manifold_k = manifold_k_neighbors
        self.random_seed = random_seed

        self.ref_predictions: Optional[np.ndarray] = None
        self.ref_uncertainties: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.ref_nn_model: Optional[NearestNeighbors] = None
        self.ref_manifold_distances: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit_reference(
        self,
        ref_predictions: np.ndarray,
        ref_uncertainties: np.ndarray,
        ref_features: Optional[np.ndarray] = None,
    ) -> "UncertaintyDriftDetector":
        """Establishes reference distributions for predictions, tree variances, and standardized manifold distances."""
        self.ref_predictions = np.asarray(ref_predictions, dtype=float)
        self.ref_uncertainties = np.asarray(ref_uncertainties, dtype=float)

        if ref_features is not None:
            rng = np.random.default_rng(self.random_seed)
            feat_arr = np.asarray(ref_features, dtype=float)
            sample_size = min(5000, len(feat_arr))
            idx = rng.choice(len(feat_arr), sample_size, replace=False)
            sub_feats = feat_arr[idx]

            self.scaler = StandardScaler()
            sub_feats_scaled = self.scaler.fit_transform(sub_feats)

            k_eff = min(self.manifold_k + 1, len(sub_feats_scaled))
            self.ref_nn_model = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
            self.ref_nn_model.fit(sub_feats_scaled)
            # Exclude self-match (distance 0 at index 0) when evaluating reference on itself
            dists, _ = self.ref_nn_model.kneighbors(sub_feats_scaled)
            self.ref_manifold_distances = dists[:, 1:].mean(axis=1) if k_eff > 1 else dists.mean(axis=1)

        self.is_fitted = True
        return self

    def evaluate(
        self,
        prod_predictions: np.ndarray,
        prod_uncertainties: np.ndarray,
        prod_features: Optional[np.ndarray] = None,
    ) -> UncertaintyReport:
        """Evaluates production batch against reference uncertainty baseline."""
        if not self.is_fitted or self.ref_predictions is None or self.ref_uncertainties is None:
            raise RuntimeError("Detector must be fitted on reference data first.")

        prod_preds = np.asarray(prod_predictions, dtype=float)
        prod_unc = np.asarray(prod_uncertainties, dtype=float)

        # 1. Prediction distribution drift
        pred_psi_res = calculate_psi_numerical(
            self.ref_predictions, prod_preds, warn_threshold=self.psi_warn, crit_threshold=self.psi_crit
        )
        pred_ks_stat, pred_ks_pval, _ = calculate_ks_test(self.ref_predictions, prod_preds)
        pred_wass = calculate_wasserstein_distance(self.ref_predictions, prod_preds)

        # 2. Epistemic uncertainty shift
        unc_psi_res = calculate_psi_numerical(
            self.ref_uncertainties, prod_unc, warn_threshold=self.psi_warn, crit_threshold=self.psi_crit
        )
        ref_mean_unc = max(1e-6, float(np.mean(self.ref_uncertainties)))
        prod_mean_unc = float(np.mean(prod_unc))
        unc_ratio = prod_mean_unc / ref_mean_unc

        # 3. Standardized Manifold distance shift
        manifold_drift = 0.0
        if self.ref_nn_model is not None and self.scaler is not None and prod_features is not None:
            rng = np.random.default_rng(self.random_seed)
            prod_feat_arr = np.asarray(prod_features, dtype=float)
            sample_size = min(3000, len(prod_feat_arr))
            idx = rng.choice(len(prod_feat_arr), sample_size, replace=False)
            prod_scaled = self.scaler.transform(prod_feat_arr[idx])

            k_query = min(self.manifold_k, self.ref_nn_model.n_samples_fit_)
            prod_dists, _ = self.ref_nn_model.kneighbors(prod_scaled, n_neighbors=k_query)
            prod_manifold_dists = prod_dists.mean(axis=1)
            manifold_drift = float(np.mean(prod_manifold_dists) / (np.mean(self.ref_manifold_distances) + 1e-6))

        # Alert level determination and justification
        justifications = []
        is_critical = False
        is_warning = False

        if pred_psi_res.psi_value >= self.psi_crit:
            is_critical = True
            justifications.append(
                f"Critical prediction distribution shift (PSI = {pred_psi_res.psi_value:.4f} >= {self.psi_crit})"
            )
        elif pred_psi_res.psi_value >= self.psi_warn:
            is_warning = True
            justifications.append(
                f"Moderate prediction distribution shift (PSI = {pred_psi_res.psi_value:.4f} >= {self.psi_warn})"
            )

        if unc_ratio >= self.unc_crit or unc_psi_res.psi_value >= self.psi_crit:
            is_critical = True
            justifications.append(
                f"Severe epistemic uncertainty inflation (Uncertainty Ratio = {unc_ratio:.2f}x, PSI = {unc_psi_res.psi_value:.4f})"
            )
        elif unc_ratio >= self.unc_warn or unc_psi_res.psi_value >= self.psi_warn:
            is_warning = True
            justifications.append(f"Elevated model uncertainty (Uncertainty Ratio = {unc_ratio:.2f}x)")

        if manifold_drift >= self.manifold_crit:
            is_critical = True
            justifications.append(f"Critical out-of-distribution manifold divergence (Distance Ratio = {manifold_drift:.2f}x)")
        elif manifold_drift >= self.manifold_warn:
            is_warning = True
            justifications.append(f"Moderate out-of-distribution manifold divergence (Distance Ratio = {manifold_drift:.2f}x)")

        if is_critical:
            status = "CRITICAL"
        elif is_warning:
            status = "WARNING"
        else:
            status = "STABLE"
            justifications.append("Prediction distribution and epistemic uncertainty within calibrated nominal bounds.")

        return UncertaintyReport(
            pred_psi=pred_psi_res.psi_value,
            pred_ks_stat=pred_ks_stat,
            pred_ks_pval=pred_ks_pval,
            pred_wasserstein=pred_wass,
            uncertainty_psi=unc_psi_res.psi_value,
            uncertainty_mean_ratio=unc_ratio,
            manifold_distance_drift=manifold_drift,
            alert_status=status,
            justification="; ".join(justifications),
        )
"""
Uncertainty-Based and Unsupervised Silent Drift Detectors.

The Label Delay Problem:
------------------------
In production machine learning systems, ground truth labels y are subject to severe latency:
- Ride-hailing / Taxi: Dispute resolution, refund processing, weekly driver accounting.
- Credit / Lending: Default status takes 6-36 months to manifest.
- Fraud / Cyber: Audit investigations and chargeback cycles take 30-90 days.

If a monitoring system relies solely on supervised performance metrics (RMSE, AUC, F1),
model failures remain completely undetected during the delay window ("Silent Failures").

Solution: Unsupervised Confidence & Manifold Proxies
----------------------------------------------------
1. Output Prediction Distribution Drift P(\hat{Y}):
   Monitors whether the model's predicted tip amounts have fundamentally shifted in shape
   or expectation, calculated via PSI and Wasserstein distance against reference predictions.

2. Ensemble Epistemic Uncertainty Distribution:
   Evaluates disagreement across individual decision trees in the gradient booster:
       sigma^2(x) = 1/T * sum_{t=1}^T (f_t(x) - f_bar(x))^2
   When inputs land in dense, well-sampled regions of the reference manifold, decision trees
   agree closely (low variance). When inputs shift out-of-distribution, trees extrapolate
   wildly along divergent split trajectories (high epistemic uncertainty).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .statistical_tests import (
    calculate_psi_numerical,
    calculate_ks_test,
    calculate_wasserstein_distance,
    PSIBinningResult,
)


@dataclass
class UncertaintyReport:
    pred_psi: float
    pred_ks_stat: float
    pred_ks_pval: float
    pred_wasserstein: float
    uncertainty_psi: float
    uncertainty_mean_ratio: float  # prod_mean_uncertainty / ref_mean_uncertainty
    manifold_distance_drift: float  # shift in distance to nearest training neighbors
    alert_status: str  # "STABLE", "WARNING", "CRITICAL"
    justification: str


class UncertaintyDriftDetector:
    """
    Monitors model prediction distribution and epistemic uncertainty to detect silent drift
    without requiring ground truth labels.
    """

    def __init__(
        self,
        psi_threshold_warning: float = 0.10,
        psi_threshold_critical: float = 0.25,
        uncertainty_ratio_warning: float = 1.35,
        uncertainty_ratio_critical: float = 1.75,
        manifold_k_neighbors: int = 10,
    ):
        self.psi_warn = psi_threshold_warning
        self.psi_crit = psi_threshold_critical
        self.unc_warn = uncertainty_ratio_warning
        self.unc_crit = uncertainty_ratio_critical
        self.manifold_k = manifold_k_neighbors

        self.ref_predictions: Optional[np.ndarray] = None
        self.ref_uncertainties: Optional[np.ndarray] = None
        self.ref_nn_model: Optional[NearestNeighbors] = None
        self.ref_manifold_distances: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit_reference(
        self,
        ref_predictions: np.ndarray,
        ref_uncertainties: np.ndarray,
        ref_features: Optional[np.ndarray] = None,
    ) -> "UncertaintyDriftDetector":
        """Establishes reference distributions for predictions, tree variances, and manifold distances."""
        self.ref_predictions = np.asarray(ref_predictions, dtype=float)
        self.ref_uncertainties = np.asarray(ref_uncertainties, dtype=float)

        if ref_features is not None:
            # Subsample for efficient neighbor searching if reference is large
            feat_arr = np.asarray(ref_features, dtype=float)
            sample_size = min(5000, len(feat_arr))
            idx = np.random.choice(len(feat_arr), sample_size, replace=False)
            sub_feats = feat_arr[idx]

            self.ref_nn_model = NearestNeighbors(n_neighbors=self.manifold_k, metric="euclidean", n_jobs=-1)
            self.ref_nn_model.fit(sub_feats)
            dists, _ = self.ref_nn_model.kneighbors(sub_feats)
            self.ref_manifold_distances = dists.mean(axis=1)

        self.is_fitted = True
        return self

    def evaluate(
        self,
        prod_predictions: np.ndarray,
        prod_uncertainties: np.ndarray,
        prod_features: Optional[np.ndarray] = None,
    ) -> UncertaintyReport:
        """Evaluates production batch against reference uncertainty baseline."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted on reference data first.")

        prod_preds = np.asarray(prod_predictions, dtype=float)
        prod_unc = np.asarray(prod_uncertainties, dtype=float)

        # 1. Prediction distribution drift
        pred_psi_res = calculate_psi_numerical(self.ref_predictions, prod_preds)
        pred_ks_stat, pred_ks_pval, _ = calculate_ks_test(self.ref_predictions, prod_preds)
        pred_wass = calculate_wasserstein_distance(self.ref_predictions, prod_preds)

        # 2. Epistemic uncertainty shift
        unc_psi_res = calculate_psi_numerical(self.ref_uncertainties, prod_unc)
        ref_mean_unc = max(1e-4, float(np.mean(self.ref_uncertainties)))
        prod_mean_unc = float(np.mean(prod_unc))
        unc_ratio = prod_mean_unc / ref_mean_unc

        # 3. Manifold distance shift
        manifold_drift = 0.0
        if self.ref_nn_model is not None and prod_features is not None:
            prod_feat_arr = np.asarray(prod_features, dtype=float)
            sample_size = min(3000, len(prod_feat_arr))
            idx = np.random.choice(len(prod_feat_arr), sample_size, replace=False)
            prod_dists, _ = self.ref_nn_model.kneighbors(prod_feat_arr[idx])
            prod_manifold_dists = prod_dists.mean(axis=1)
            manifold_drift = float(np.mean(prod_manifold_dists) / (np.mean(self.ref_manifold_distances) + 1e-4))

        # Alert level determination and justification
        justifications = []
        is_critical = False
        is_warning = False

        if pred_psi_res.psi_value >= self.psi_crit:
            is_critical = True
            justifications.append(f"Critical prediction distribution shift (PSI = {pred_psi_res.psi_value:.4f} >= {self.psi_crit})")
        elif pred_psi_res.psi_value >= self.psi_warn:
            is_warning = True
            justifications.append(f"Moderate prediction distribution shift (PSI = {pred_psi_res.psi_value:.4f} >= {self.psi_warn})")

        if unc_ratio >= self.unc_crit or unc_psi_res.psi_value >= self.psi_crit:
            is_critical = True
            justifications.append(f"Severe epistemic uncertainty inflation (Uncertainty Ratio = {unc_ratio:.2f}x, PSI = {unc_psi_res.psi_value:.4f})")
        elif unc_ratio >= self.unc_warn or unc_psi_res.psi_value >= self.psi_warn:
            is_warning = True
            justifications.append(f"Elevated model uncertainty (Uncertainty Ratio = {unc_ratio:.2f}x)")

        if manifold_drift >= 1.50:
            justifications.append(f"Out-of-distribution manifold divergence (Distance Ratio = {manifold_drift:.2f}x)")

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

"""
Statistical Drift Detection Engine.

Statistical Foundations:
-----------------------
1. Population Stability Index (PSI):
   Measures divergence between two probability distributions P (reference) and Q (production).
   Formula:
       PSI = sum_{i=1}^B (P_i - Q_i) * ln( (P_i + eps) / (Q_i + eps) )
   Properties:
   - Symmetric relative entropy approximation (symmetrized Kullback-Leibler divergence).
   - Scale-invariant because it operates on binned relative frequencies.
   - Robust to non-normal, multimodal, and skewed distributions.
   - Industry Standard Benchmarks:
       * PSI < 0.10: No significant distribution change (Stable).
       * 0.10 <= PSI < 0.25: Moderate shift; warrants monitoring/warning.
       * PSI >= 0.25: Actionable severe drift; triggers root-cause attribution.

2. Kolmogorov-Smirnov (KS) Two-Sample Test:
   Non-parametric test comparing empirical cumulative distribution functions (eCDFs).
   Statistic:
       D = sup_x | F_ref(x) - F_prod(x) |
   Properties:
   - Sensitive to shape, location, and dispersion differences across the whole continuous domain.
   - Requires no prior binning assumptions.
   - Problem in Production: With large sample sizes (N > 10,000), even minuscule, practically
     irrelevant shifts produce p-values < 1e-15. Hence, we report both the test statistic D
     and the p-value.

3. Multiple Hypothesis Testing Correction (Benjamini-Hochberg FDR):
   When monitoring M features concurrently at alpha = 0.01, the naive Family-Wise Error Rate
   FWER = 1 - (1 - alpha)^M explodes (e.g. for M=16, FWER ≈ 15%).
   We apply the Benjamini-Hochberg False Discovery Rate (FDR) procedure to control the expected
   proportion of false alarms.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PSIBinningResult:
    psi_value: float
    bin_edges: np.ndarray
    ref_proportions: np.ndarray
    prod_proportions: np.ndarray
    bin_contributions: np.ndarray
    status: str  # "STABLE", "WARNING", "CRITICAL"


def calculate_psi_numerical(
    reference: Union[np.ndarray, pd.Series],
    production: Union[np.ndarray, pd.Series],
    n_bins: int = 10,
    epsilon: float = 1e-5,
    bin_edges: Optional[np.ndarray] = None,
) -> PSIBinningResult:
    """
    Computes Population Stability Index (PSI) for continuous numerical features.
    
    Why Quantile Binning?
    Quantile (equal-frequency) binning on the reference set ensures that each baseline
    bucket receives exactly ~1/B of the mass, preventing bucket starvation in the tail
    and maximizing statistical test power.
    """
    ref_clean = np.asarray(reference, dtype=float)
    prod_clean = np.asarray(production, dtype=float)

    ref_clean = ref_clean[~np.isnan(ref_clean)]
    prod_clean = prod_clean[~np.isnan(prod_clean)]

    if len(ref_clean) == 0 or len(prod_clean) == 0:
        return PSIBinningResult(0.0, np.array([]), np.array([]), np.array([]), np.array([]), "STABLE")

    # If reference is constant, check if production matches
    if np.all(ref_clean == ref_clean[0]):
        prod_same = np.mean(prod_clean == ref_clean[0])
        psi = 0.0 if prod_same > 0.99 else 1.0
        status = "CRITICAL" if psi >= 0.25 else "STABLE"
        return PSIBinningResult(psi, np.array([ref_clean[0]]), np.array([1.0]), np.array([prod_same]), np.array([psi]), status)

    # Derive bin edges from reference quantiles if not provided
    if bin_edges is None:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(ref_clean, quantiles)
        # Ensure strictly monotonic bin edges for duplicate quantile values
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([ref_clean.min() - 1e-3, ref_clean.max() + 1e-3])
        else:
            bin_edges[0] -= 1e-5
            bin_edges[-1] += 1e-5

    # Compute empirical bucket counts
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    prod_counts, _ = np.histogram(prod_clean, bins=bin_edges)

    # Convert to probability mass distributions
    ref_props = ref_counts / len(ref_clean)
    prod_props = prod_counts / len(prod_clean)

    # Apply Laplace smoothing to prevent division by zero or log(0)
    ref_smooth = np.clip(ref_props, epsilon, None)
    prod_smooth = np.clip(prod_props, epsilon, None)

    # Re-normalize smoothed distributions
    ref_smooth = ref_smooth / np.sum(ref_smooth)
    prod_smooth = prod_smooth / np.sum(prod_smooth)

    # Symmetric divergence calculation
    bin_contribs = (prod_smooth - ref_smooth) * np.log(prod_smooth / ref_smooth)
    psi_value = float(np.sum(bin_contribs))

    # Alert classification based on risk guidelines
    if psi_value < 0.10:
        status = "STABLE"
    elif psi_value < 0.25:
        status = "WARNING"
    else:
        status = "CRITICAL"

    return PSIBinningResult(
        psi_value=psi_value,
        bin_edges=bin_edges,
        ref_proportions=ref_props,
        prod_proportions=prod_props,
        bin_contributions=bin_contribs,
        status=status,
    )


def calculate_psi_categorical(
    reference: Union[np.ndarray, pd.Series],
    production: Union[np.ndarray, pd.Series],
    epsilon: float = 1e-5,
) -> PSIBinningResult:
    """
    Computes Population Stability Index (PSI) for discrete/categorical features.
    Builds discrete category union and tracks unseen category emergence.
    """
    ref_s = pd.Series(reference).dropna()
    prod_s = pd.Series(production).dropna()

    if len(ref_s) == 0 or len(prod_s) == 0:
        return PSIBinningResult(0.0, np.array([]), np.array([]), np.array([]), np.array([]), "STABLE")

    all_categories = sorted(list(set(ref_s.unique()).union(set(prod_s.unique()))))

    ref_counts = ref_s.value_counts(normalize=True).to_dict()
    prod_counts = prod_s.value_counts(normalize=True).to_dict()

    ref_props = np.array([ref_counts.get(cat, 0.0) for cat in all_categories])
    prod_props = np.array([prod_counts.get(cat, 0.0) for cat in all_categories])

    ref_smooth = np.clip(ref_props, epsilon, None)
    prod_smooth = np.clip(prod_props, epsilon, None)
    ref_smooth = ref_smooth / np.sum(ref_smooth)
    prod_smooth = prod_smooth / np.sum(prod_smooth)

    bin_contribs = (prod_smooth - ref_smooth) * np.log(prod_smooth / ref_smooth)
    psi_value = float(np.sum(bin_contribs))

    if psi_value < 0.10:
        status = "STABLE"
    elif psi_value < 0.25:
        status = "WARNING"
    else:
        status = "CRITICAL"

    return PSIBinningResult(
        psi_value=psi_value,
        bin_edges=np.array(all_categories),
        ref_proportions=ref_props,
        prod_proportions=prod_props,
        bin_contributions=bin_contribs,
        status=status,
    )


def calculate_ks_test(
    reference: Union[np.ndarray, pd.Series],
    production: Union[np.ndarray, pd.Series],
) -> Tuple[float, float, str]:
    """
    Two-sample Kolmogorov-Smirnov test for continuous distributions.
    Returns:
        (ks_statistic D, p_value, status)
    """
    ref_clean = np.asarray(reference, dtype=float)
    prod_clean = np.asarray(production, dtype=float)
    ref_clean = ref_clean[~np.isnan(ref_clean)]
    prod_clean = prod_clean[~np.isnan(prod_clean)]

    if len(ref_clean) == 0 or len(prod_clean) == 0:
        return 0.0, 1.0, "STABLE"

    ks_stat, p_val = stats.ks_2samp(ref_clean, prod_clean)

    # We evaluate both p_val and effect size D:
    # Small p_val (< 0.01) with D >= 0.05 indicates substantial practical drift.
    if p_val < 0.001 and ks_stat >= 0.10:
        status = "CRITICAL"
    elif p_val < 0.01 and ks_stat >= 0.04:
        status = "WARNING"
    else:
        status = "STABLE"

    return float(ks_stat), float(p_val), status


def calculate_wasserstein_distance(
    reference: Union[np.ndarray, pd.Series],
    production: Union[np.ndarray, pd.Series],
) -> float:
    """
    Computes 1st Wasserstein (Earth Mover's) Distance:
        W1(u, v) = integral | U(t) - V(t) | dt
    Provides an interpretable, units-preserving metric of distribution shift.
    """
    ref_clean = np.asarray(reference, dtype=float)
    prod_clean = np.asarray(production, dtype=float)
    ref_clean = ref_clean[~np.isnan(ref_clean)]
    prod_clean = prod_clean[~np.isnan(prod_clean)]

    if len(ref_clean) == 0 or len(prod_clean) == 0:
        return 0.0

    return float(stats.wasserstein_distance(ref_clean, prod_clean))


def apply_benjamini_hochberg_correction(
    p_values: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, Tuple[bool, float]]:
    """
    Applies the Benjamini-Hochberg (BH) step-up procedure to control False Discovery Rate (FDR).
    
    Algorithm:
    1. Sort M tests in ascending order of p-value: p_(1) <= p_(2) <= ... <= p_(M).
    2. Find largest index k such that p_(k) <= (k / M) * alpha.
    3. Reject null hypotheses for all tests i = 1, ..., k.
    4. Compute adjusted p-values (q-values): q_(i) = min_{j >= i} ( (M / j) * p_(j) ).
    
    Returns:
        Dict[feature_name, (is_significant_rejection, adjusted_q_value)]
    """
    features = list(p_values.keys())
    m = len(features)
    if m == 0:
        return {}

    sorted_pairs = sorted([(feat, p_values[feat]) for feat in features], key=lambda x: x[1])
    sorted_features = [x[0] for x in sorted_pairs]
    raw_p = np.array([x[1] for x in sorted_pairs])

    # Compute step-up thresholds: (i / m) * alpha for i in 1..m
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * alpha

    # Find maximum rank satisfying criterion
    significant_ranks = np.where(raw_p <= thresholds)[0]
    max_k = significant_ranks[-1] if len(significant_ranks) > 0 else -1

    # Compute adjusted q-values
    q_values = np.zeros(m)
    current_min = 1.0
    for i in range(m - 1, -1, -1):
        q = (m / (i + 1)) * raw_p[i]
        current_min = min(current_min, q)
        q_values[i] = min(1.0, current_min)

    results: Dict[str, Tuple[bool, float]] = {}
    for idx, feat in enumerate(sorted_features):
        is_sig = idx <= max_k
        results[feat] = (bool(is_sig), float(q_values[idx]))

    return results

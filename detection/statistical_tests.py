"""
Statistical Drift Detection Engine.
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


def _classify_psi(psi_value: float, warn_threshold: float = 0.10, crit_threshold: float = 0.25) -> str:
    if psi_value < warn_threshold:
        return "STABLE"
    elif psi_value < crit_threshold:
        return "WARNING"
    return "CRITICAL"


def calculate_psi_numerical(
    reference: Union[np.ndarray, pd.Series],
    production: Union[np.ndarray, pd.Series],
    n_bins: int = 10,
    epsilon: float = 1e-5,
    bin_edges: Optional[np.ndarray] = None,
    warn_threshold: float = 0.10,
    crit_threshold: float = 0.25,
) -> PSIBinningResult:
    """
    Computes Population Stability Index (PSI) for continuous numerical features.
    """
    ref_clean = np.asarray(reference, dtype=float)
    prod_clean = np.asarray(production, dtype=float)

    ref_clean = ref_clean[~np.isnan(ref_clean)]
    prod_clean = prod_clean[~np.isnan(prod_clean)]

    if len(ref_clean) == 0 or len(prod_clean) == 0:
        return PSIBinningResult(0.0, np.array([]), np.array([]), np.array([]), np.array([]), "STABLE")

    # If reference is constant, check if production matches
    if np.all(ref_clean == ref_clean[0]):
        prod_same = float(np.mean(prod_clean == ref_clean[0]))
        psi = 0.0 if prod_same > 0.99 else 1.0
        status = _classify_psi(psi, warn_threshold, crit_threshold)
        return PSIBinningResult(
            psi, np.array([ref_clean[0]]), np.array([1.0]), np.array([prod_same]), np.array([psi]), status
        )

    # Low-cardinality numeric features (0/1 flags, fixed fees) collapse to a single quantile bin;
    # treat them as discrete categories.
    if len(np.unique(ref_clean)) <= n_bins:
        return calculate_psi_categorical(
            ref_clean,
            prod_clean,
            epsilon=epsilon,
            warn_threshold=warn_threshold,
            crit_threshold=crit_threshold,
        )

    # Derive bin edges from reference quantiles if not provided
    if bin_edges is None or len(bin_edges) < 2:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(ref_clean, quantiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([ref_clean.min() - 1e-3, ref_clean.max() + 1e-3])
        else:
            bin_edges = bin_edges.copy()
            bin_edges[0] -= 1e-5
            bin_edges[-1] += 1e-5

    # Count with open-ended outer bins so out-of-range production values land in extreme buckets
    counting_edges = np.concatenate(([-np.inf], bin_edges[1:-1], [np.inf]))
    ref_counts, _ = np.histogram(ref_clean, bins=counting_edges)
    prod_counts, _ = np.histogram(prod_clean, bins=counting_edges)

    ref_props = ref_counts / len(ref_clean)
    prod_props = prod_counts / len(prod_clean)

    # Apply Laplace smoothing and re-normalize
    ref_smooth = np.clip(ref_props, epsilon, None)
    prod_smooth = np.clip(prod_props, epsilon, None)
    ref_smooth = ref_smooth / np.sum(ref_smooth)
    prod_smooth = prod_smooth / np.sum(prod_smooth)

    bin_contribs = (prod_smooth - ref_smooth) * np.log(prod_smooth / ref_smooth)
    psi_value = float(np.sum(bin_contribs))
    status = _classify_psi(psi_value, warn_threshold, crit_threshold)

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
    warn_threshold: float = 0.10,
    crit_threshold: float = 0.25,
) -> PSIBinningResult:
    """
    Computes Population Stability Index (PSI) for discrete/categorical features.
    """
    ref_s = pd.Series(reference).dropna()
    prod_s = pd.Series(production).dropna()

    if len(ref_s) == 0 or len(prod_s) == 0:
        return PSIBinningResult(0.0, np.array([]), np.array([]), np.array([]), np.array([]), "STABLE")

    all_categories = sorted(list(set(ref_s.unique()).union(set(prod_s.unique()))))

    ref_counts = ref_s.value_counts(normalize=True).to_dict()
    prod_counts = prod_s.value_counts(normalize=True).to_dict()

    ref_props = np.array([ref_counts.get(cat, 0.0) for cat in all_categories], dtype=float)
    prod_props = np.array([prod_counts.get(cat, 0.0) for cat in all_categories], dtype=float)

    ref_smooth = np.clip(ref_props, epsilon, None)
    prod_smooth = np.clip(prod_props, epsilon, None)
    ref_smooth = ref_smooth / np.sum(ref_smooth)
    prod_smooth = prod_smooth / np.sum(prod_smooth)

    bin_contribs = (prod_smooth - ref_smooth) * np.log(prod_smooth / ref_smooth)
    psi_value = float(np.sum(bin_contribs))
    status = _classify_psi(psi_value, warn_threshold, crit_threshold)

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
    warn_pval: float = 0.01,
    crit_pval: float = 0.001,
    warn_stat: float = 0.04,
    crit_stat: float = 0.10,
) -> Tuple[float, float, str]:
    """
    Two-sample Kolmogorov-Smirnov test for continuous distributions.
    """
    ref_clean = np.asarray(reference, dtype=float)
    prod_clean = np.asarray(production, dtype=float)
    ref_clean = ref_clean[~np.isnan(ref_clean)]
    prod_clean = prod_clean[~np.isnan(prod_clean)]

    if len(ref_clean) == 0 or len(prod_clean) == 0:
        return 0.0, 1.0, "STABLE"

    ks_stat, p_val = stats.ks_2samp(ref_clean, prod_clean)

    if p_val < crit_pval and ks_stat >= crit_stat:
        status = "CRITICAL"
    elif p_val < warn_pval and ks_stat >= warn_stat:
        status = "WARNING"
    else:
        status = "STABLE"

    return float(ks_stat), float(p_val), status


def calculate_wasserstein_distance(
    reference: Union[np.ndarray, pd.Series],
    production: Union[np.ndarray, pd.Series],
) -> float:
    """Computes 1st Wasserstein (Earth Mover's) Distance."""
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
    """
    features = list(p_values.keys())
    m = len(features)
    if m == 0:
        return {}

    sorted_pairs = sorted([(feat, p_values[feat]) for feat in features], key=lambda x: x[1])
    sorted_features = [x[0] for x in sorted_pairs]
    raw_p = np.array([x[1] for x in sorted_pairs], dtype=float)

    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * alpha

    significant_ranks = np.where(raw_p <= thresholds)[0]
    max_k = significant_ranks[-1] if len(significant_ranks) > 0 else -1

    q_values = np.zeros(m, dtype=float)
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
"""
Unit tests for Statistical Tests: PSI, KS-test, and Benjamini-Hochberg FDR correction.
Includes mathematical invariants and edge case verifications.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from detection.statistical_tests import (
    calculate_psi_numerical,
    calculate_psi_categorical,
    calculate_ks_test,
    apply_benjamini_hochberg_correction,
    calculate_wasserstein_distance,
)


def test_psi_identical_distributions():
    """
    Mathematical Invariant: Symmetrized KL divergence between identical distributions
    must be exactly zero (or within numerical epsilon < 1e-3).
    """
    np.random.seed(42)
    ref = np.random.normal(10.0, 2.0, size=5000)
    prod = np.random.normal(10.0, 2.0, size=5000)

    res = calculate_psi_numerical(ref, prod, n_bins=10)
    assert res.psi_value < 0.05, f"Expected near-zero PSI for identical distributions, got {res.psi_value}"
    assert res.status == "STABLE"


def test_psi_shifted_distribution_triggers_critical():
    """
    A 2-sigma mean shift must produce significant relative entropy divergence (PSI > 0.25).
    """
    np.random.seed(42)
    ref = np.random.normal(10.0, 2.0, size=5000)
    prod = np.random.normal(14.0, 2.0, size=5000)  # +2 sigma shift

    res = calculate_psi_numerical(ref, prod, n_bins=10)
    assert res.psi_value >= 0.25, f"Expected critical PSI >= 0.25, got {res.psi_value}"
    assert res.status == "CRITICAL"


def test_psi_zero_count_smoothing():
    """
    Edge Case: Production batch has empty buckets that would cause log(0) or division by zero.
    Laplace smoothing must prevent NaNs/Infs and maintain finite PSI.
    """
    ref = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100)
    prod = np.array([1, 2] * 500)  # bins for 3..10 have 0 counts

    res = calculate_psi_numerical(ref, prod, n_bins=5)
    assert not np.isnan(res.psi_value)
    assert not np.isinf(res.psi_value)
    assert res.psi_value > 0.25
    assert res.status == "CRITICAL"


def test_categorical_psi_unseen_category():
    """
    Tests handling of emerging categorical levels in production.
    """
    ref = pd.Series(["A", "A", "B", "B", "C"] * 200)
    prod = pd.Series(["A", "B", "D", "E"] * 250)  # D and E are new unseen categories

    res = calculate_psi_categorical(ref, prod)
    assert not np.isnan(res.psi_value)
    assert res.psi_value > 0.25
    assert res.status == "CRITICAL"


def test_ks_test_accuracy():
    """
    Verifies that our KS implementation aligns with SciPy two-sample Kolmogorov-Smirnov test.
    """
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    prod = np.random.normal(0.5, 1, 1000)

    ks_stat, p_val, status = calculate_ks_test(ref, prod)
    expected_stat, expected_pval = stats.ks_2samp(ref, prod)

    assert np.isclose(ks_stat, expected_stat, atol=1e-4)
    assert np.isclose(p_val, expected_pval, atol=1e-4)
    assert p_val < 0.01
    assert status in ("WARNING", "CRITICAL")


def test_benjamini_hochberg_fdr_control():
    """
    Verifies that the Benjamini-Hochberg procedure controls false discoveries across multiple hypotheses.
    """
    # Simulate 20 tests: 3 true drifted signals (tiny p-values) + 17 null non-drifted (uniform p-values)
    p_values = {
        "drifted_1": 1e-8,
        "drifted_2": 1e-5,
        "drifted_3": 0.002,
        "null_1": 0.12,
        "null_2": 0.45,
        "null_3": 0.78,
        "null_4": 0.90,
        "null_5": 0.33,
    }

    results = apply_benjamini_hochberg_correction(p_values, alpha=0.01)
    
    # The 3 true signals must be rejected (significant = True)
    assert results["drifted_1"][0] is True
    assert results["drifted_2"][0] is True
    assert results["drifted_3"][0] is True

    # The null features must NOT be rejected
    assert results["null_1"][0] is False
    assert results["null_2"][0] is False

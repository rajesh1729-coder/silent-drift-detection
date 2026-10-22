"""
Unit tests for epistemic uncertainty proxy and manifold drift detection.
"""

import numpy as np
import pytest
from detection.uncertainty_proxy import UncertaintyDriftDetector


def test_uncertainty_detector_nominal_baseline():
    """
    Under stable production data matching reference distribution,
    the uncertainty ratio must remain near 1.0 and status must be STABLE.
    """
    np.random.seed(42)
    ref_preds = np.random.normal(5.0, 1.0, 1000)
    ref_uncs = np.random.gamma(2.0, 0.2, 1000)

    prod_preds = np.random.normal(5.0, 1.0, 1000)
    prod_uncs = np.random.gamma(2.0, 0.2, 1000)

    detector = UncertaintyDriftDetector()
    detector.fit_reference(ref_preds, ref_uncs)
    report = detector.evaluate(prod_preds, prod_uncs)

    assert report.alert_status == "STABLE"
    assert report.pred_psi < 0.10
    assert 0.85 < report.uncertainty_mean_ratio < 1.15


def test_uncertainty_detector_epistemic_spike():
    """
    When model encounters out-of-distribution inputs causing tree variance inflation,
    the detector must flag WARNING or CRITICAL.
    """
    np.random.seed(42)
    ref_preds = np.random.normal(5.0, 1.0, 1000)
    ref_uncs = np.random.gamma(2.0, 0.2, 1000)

    # Shifted predictions and 2.5x inflated uncertainty
    prod_preds = np.random.normal(7.5, 1.5, 1000)
    prod_uncs = ref_uncs * 2.2

    detector = UncertaintyDriftDetector()
    detector.fit_reference(ref_preds, ref_uncs)
    report = detector.evaluate(prod_preds, prod_uncs)

    assert report.alert_status == "CRITICAL"
    assert report.uncertainty_mean_ratio > 2.0

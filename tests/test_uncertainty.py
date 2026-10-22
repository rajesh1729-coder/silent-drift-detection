"""
Unit tests for epistemic uncertainty proxy and manifold drift detection.
"""

import numpy as np
from detection.uncertainty_proxy import UncertaintyDriftDetector


def test_uncertainty_detector_nominal_baseline():
    """
    Under stable production data matching reference distribution,
    the uncertainty ratio must remain near 1.0 and status must be STABLE.
    """
    rng = np.random.default_rng(42)
    ref_preds = rng.normal(5.0, 1.0, 1000)
    ref_uncs = rng.gamma(2.0, 0.2, 1000)

    prod_preds = rng.normal(5.0, 1.0, 1000)
    prod_uncs = rng.gamma(2.0, 0.2, 1000)

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
    rng = np.random.default_rng(42)
    ref_preds = rng.normal(5.0, 1.0, 1000)
    ref_uncs = rng.gamma(2.0, 0.2, 1000)

    prod_preds = rng.normal(7.5, 1.5, 1000)
    prod_uncs = ref_uncs * 2.2

    detector = UncertaintyDriftDetector()
    detector.fit_reference(ref_preds, ref_uncs)
    report = detector.evaluate(prod_preds, prod_uncs)

    assert report.alert_status == "CRITICAL"
    assert report.uncertainty_mean_ratio > 2.0


def test_manifold_drift_triggers_alert_status():
    """Severe k-NN manifold shift must trigger CRITICAL even if predictions and uncertainties are stable."""
    rng = np.random.default_rng(42)
    ref_preds = rng.normal(5.0, 1.0, 500)
    ref_uncs = rng.gamma(2.0, 0.2, 500)
    ref_feats = rng.normal(0.0, 1.0, size=(500, 4))

    prod_feats = rng.normal(10.0, 1.0, size=(500, 4))

    detector = UncertaintyDriftDetector()
    detector.fit_reference(ref_preds, ref_uncs, ref_feats)
    report = detector.evaluate(ref_preds, ref_uncs, prod_feats)

    assert report.manifold_distance_drift > 2.25
    assert report.alert_status == "CRITICAL"
"""Tests for the DriftDetector decision logic (status rules and FDR gating)."""

import numpy as np
import pandas as pd
import pytest

from detection.detector import DriftDetector, combine_feature_status


@pytest.mark.parametrize(
    "psi_status, ks_status, fdr_sig, expected",
    [
        ("STABLE", "STABLE", True, "STABLE"),
        ("STABLE", "WARNING", True, "WARNING"),
        ("STABLE", "CRITICAL", True, "CRITICAL"),
        ("STABLE", "WARNING", False, "STABLE"),
        ("STABLE", "CRITICAL", False, "STABLE"),
        ("WARNING", "STABLE", False, "WARNING"),
        ("CRITICAL", "STABLE", False, "CRITICAL"),
        ("WARNING", "CRITICAL", True, "CRITICAL"),
        ("WARNING", "CRITICAL", False, "WARNING"),
    ],
)
def test_combine_feature_status_rules(psi_status, ks_status, fdr_sig, expected):
    assert combine_feature_status(psi_status, ks_status, fdr_sig) == expected


def _make_frames(n=20000, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    ref = pd.DataFrame({"x": rng.normal(0, 1, n)})
    prod = pd.DataFrame({"x": rng.normal(shift, 1, n)})
    return ref, prod


def test_ks_only_signal_is_dropped_when_fdr_rejects_nothing():
    """A KS-only warning must vanish when BH is so strict that nothing is significant."""
    ref, prod = _make_frames(shift=0.12)

    lenient = DriftDetector(["x"], [], ks_alpha=0.01).fit_reference(ref)
    lenient_rep = lenient.inspect_window(prod)
    m = lenient_rep.feature_metrics["x"]
    assert m.psi_status == "STABLE"
    assert m.ks_status != "STABLE"
    assert m.status == m.ks_status

    strict = DriftDetector(["x"], [], ks_alpha=1e-300).fit_reference(ref)
    strict_rep = strict.inspect_window(prod)
    assert strict_rep.feature_metrics["x"].ks_fdr_significant is False
    assert strict_rep.feature_metrics["x"].status == "STABLE"
    assert strict_rep.overall_status == "NORMAL"


def test_clean_window_is_normal():
    ref, prod = _make_frames(shift=0.0, seed=1)
    det = DriftDetector(["x"], []).fit_reference(ref)
    assert det.inspect_window(prod).overall_status == "NORMAL"


def test_summary_names_the_critical_features_and_has_no_hardcoded_psi_claim():
    ref, prod = _make_frames(shift=3.0)
    det = DriftDetector(["x"], []).fit_reference(ref)
    rep = det.inspect_window(prod)
    assert rep.overall_status == "CRITICAL"
    assert "x" in rep.summary
    assert "PSI >= 0.25" not in rep.summary


def test_custom_psi_thresholds_are_respected():
    """Custom PSI warning and critical thresholds must override 0.10/0.25 defaults."""
    ref, prod = _make_frames(shift=0.25, seed=10)
    strict_det = DriftDetector(
        ["x"], [], psi_threshold_warning=0.01, psi_threshold_critical=0.03
    ).fit_reference(ref)
    rep = strict_det.inspect_window(prod)
    assert rep.feature_metrics["x"].psi_status == "CRITICAL"
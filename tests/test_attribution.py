"""
Unit tests for Root-Cause Attribution: SHAP Additivity Invariant and Causal Ablation.
"""

import numpy as np
import pandas as pd
import pytest

from data.download import generate_synthetic_taxi_fallback
from data.preprocess import prepare_drift_datasets
from data.model import BaselineTaxiModel
from attribution.shap_attribution import SHAPDriftAttributor
from attribution.causal_ablation import CausalAblationAttributor
from attribution.ranker import DriftAttributionEngine


@pytest.fixture(scope="module")
def prepared_model_and_data():
    raw_windows = generate_synthetic_taxi_fallback(n_samples_per_window=2000, n_windows=2)
    pipeline, proc_windows = prepare_drift_datasets(raw_windows)
    
    ref_df = proc_windows["W0_reference"]
    prod_df = proc_windows["W1_prod_month_1"]

    model = BaselineTaxiModel(n_estimators=30, max_depth=4)
    model.fit(ref_df)

    return model, ref_df, prod_df


def test_shap_additivity_invariant(prepared_model_and_data):
    """
    Mathematical Invariant: Sum of mean SHAP value shifts across all features
    must match the empirical shift in model expected predictions:
        Delta E[f(X)] ≈ sum_j Delta bar{phi}_j
    """
    model, ref_df, prod_df = prepared_model_and_data

    attributor = SHAPDriftAttributor(model, max_samples=500)
    attributor.fit_reference(ref_df)
    res = attributor.attribute(prod_df)

    sum_delta_shaps = sum(res.delta_mean_shaps.values())
    expected_delta_pred = res.delta_expected_pred

    assert np.isclose(sum_delta_shaps, expected_delta_pred, atol=0.05), (
        f"SHAP additivity violated: sum(Delta phi)={sum_delta_shaps:.4f} vs Delta E[y]={expected_delta_pred:.4f}"
    )


def test_causal_ablation_reduces_drift(prepared_model_and_data):
    """
    Causal Invariant: Performing counterfactual quantile mapping intervention
    on a feature must alter the drifted prediction mean and produce finite recovery scores.
    """
    model, ref_df, prod_df = prepared_model_and_data

    # Create synthetic severe shift on fare_amount
    drifted_df = prod_df.copy()
    drifted_df["fare_amount"] = drifted_df["fare_amount"] + 25.0

    causal_attributor = CausalAblationAttributor(model, max_samples=500)
    causal_attributor.fit_reference(ref_df)
    res = causal_attributor.attribute(drifted_df, candidate_features=["fare_amount", "trip_distance"])

    # fare_amount should have dominant prediction recovery
    assert "fare_amount" in res.prediction_recovery_scores
    assert res.prediction_recovery_scores["fare_amount"] > res.prediction_recovery_scores["trip_distance"]


def test_unified_attribution_engine(prepared_model_and_data):
    """
    Verifies that the unified engine produces consistent rankings and summaries.
    """
    model, ref_df, prod_df = prepared_model_and_data

    engine = DriftAttributionEngine(model, max_samples=500)
    engine.fit_reference(ref_df)
    report = engine.attribute_drift(prod_df)

    assert report.top_root_cause_feature in model.features
    assert len(report.top_3_root_causes) == 3
    assert len(report.feature_summaries) == len(model.features)

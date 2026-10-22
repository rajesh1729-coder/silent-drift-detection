"""
Synthetic Drift Injector for Ground-Truth Controlled Experiments.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DriftType(Enum):
    MEAN_SHIFT = "mean_shift"
    VARIANCE_SCALE = "variance_scale"
    NONLINEAR_TRANSFORM = "nonlinear_transform"
    CATEGORICAL_SWAP = "categorical_swap"
    OUTLIER_CONTAMINATION = "outlier_contamination"


@dataclass
class InjectedDriftScenario:
    scenario_id: str
    target_feature: str
    drift_type: DriftType
    magnitude: float
    description: str
    drifted_df: pd.DataFrame
    clean_reference_df: pd.DataFrame


class SyntheticDriftInjector:
    """
    Applies calibrated, reproducible distribution transformations to specific features
    in a clean held-out evaluation dataset.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def inject_drift(
        self,
        clean_df: pd.DataFrame,
        target_feature: str,
        drift_type: DriftType = DriftType.MEAN_SHIFT,
        magnitude: float = 2.0,
        scenario_id: Optional[str] = None,
    ) -> InjectedDriftScenario:
        """
        Injects a known mathematical drift into target_feature of clean_df.
        """
        if target_feature not in clean_df.columns:
            raise ValueError(f"Feature '{target_feature}' not found in input dataframe.")

        drifted = clean_df.copy()
        raw_vals = drifted[target_feature].to_numpy().copy()
        n_samples = len(raw_vals)

        is_num = np.issubdtype(raw_vals.dtype, np.number)
        std_val = float(np.std(raw_vals)) if is_num else 1.0
        mean_val = float(np.mean(raw_vals)) if is_num else 0.0
        lower_bound = 0.0 if (is_num and np.min(raw_vals) >= 0.0) else None

        if drift_type == DriftType.MEAN_SHIFT:
            shift_amount = magnitude * (std_val if std_val > 1e-4 else 1.0)
            shifted = raw_vals + shift_amount
            drifted[target_feature] = np.clip(shifted, lower_bound, None) if lower_bound is not None else shifted
            desc = f"Mean shift on '{target_feature}' by +{magnitude:.2f} sigma (+{shift_amount:.2f})"

        elif drift_type == DriftType.VARIANCE_SCALE:
            scaled = mean_val + magnitude * (raw_vals - mean_val)
            drifted[target_feature] = np.clip(scaled, lower_bound, None) if lower_bound is not None else scaled
            desc = f"Variance scaling on '{target_feature}' by {magnitude:.2f}x"

        elif drift_type == DriftType.NONLINEAR_TRANSFORM:
            span = raw_vals.max() - raw_vals.min()
            normed = (raw_vals - raw_vals.min()) / (span + 1e-5)
            warped = (normed ** magnitude) * span + raw_vals.min()
            drifted[target_feature] = warped
            desc = f"Nonlinear power transform on '{target_feature}' with exponent {magnitude:.2f}"

        elif drift_type == DriftType.CATEGORICAL_SWAP:
            cats = np.unique(raw_vals)
            if len(cats) > 1:
                dominant_cat = pd.Series(raw_vals).mode()[0]
                other_cats = [c for c in cats if c != dominant_cat]
                mask = (raw_vals == dominant_cat) & (
                    self.rng.uniform(0, 1, size=n_samples) < min(0.9, magnitude * 0.4)
                )
                raw_vals[mask] = self.rng.choice(other_cats, size=int(np.sum(mask)))
                drifted[target_feature] = raw_vals
            desc = f"Categorical distribution shift on '{target_feature}'"

        elif drift_type == DriftType.OUTLIER_CONTAMINATION:
            n_outliers = int(n_samples * min(0.30, 0.05 * magnitude))
            outlier_idx = self.rng.choice(n_samples, size=n_outliers, replace=False)
            extreme_val = mean_val + (magnitude * 4.0 * std_val)
            raw_vals[outlier_idx] = extreme_val
            drifted[target_feature] = raw_vals
            desc = f"Outlier contamination on '{target_feature}' with {n_outliers} extreme samples"

        sc_id = scenario_id or f"{target_feature}_{drift_type.value}_mag{magnitude:.1f}"

        return InjectedDriftScenario(
            scenario_id=sc_id,
            target_feature=target_feature,
            drift_type=drift_type,
            magnitude=magnitude,
            description=desc,
            drifted_df=drifted,
            clean_reference_df=clean_df,
        )
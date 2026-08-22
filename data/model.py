"""
Baseline Model Training and Inference with Uncertainty Estimation.
"""

import os
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from .preprocess import ALL_MODEL_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)


class BaselineTaxiModel:
    """
    Production LightGBM Regressor for predicting NYC Taxi Tip Amounts ($).
    Equipped with tree-ensemble variance estimators for silent/uncertainty drift detection.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        target_col: str = TARGET_COLUMN,
        n_estimators: int = 120,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        self.features = features or ALL_MODEL_FEATURES
        self.target_col = target_col
        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.model: Optional[lgb.LGBMRegressor] = None
        self.reference_metrics: Dict[str, float] = {}
        self.reference_predictions: Optional[np.ndarray] = None
        self.reference_residuals: Optional[np.ndarray] = None
        self.reference_uncertainties: Optional[np.ndarray] = None

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "BaselineTaxiModel":
        """Trains LightGBM model on reference features and records baseline metrics."""
        X_train = train_df[self.features]
        y_train = train_df[self.target_col]

        if val_df is None:
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = val_df[self.features], val_df[self.target_col]

        logger.info("Training LightGBM model on %d samples with %d features...", len(X_tr), len(self.features))
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        )

        # Baseline evaluation on validation split
        val_preds = self.model.predict(X_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        val_mae = float(mean_absolute_error(y_val, val_preds))
        val_r2 = float(r2_score(y_val, val_preds))

        self.reference_metrics = {
            "rmse": val_rmse,
            "mae": val_mae,
            "r2": val_r2,
            "mean_tip": float(np.mean(y_val)),
            "std_tip": float(np.std(y_val)),
        }
        self.reference_predictions = val_preds
        self.reference_residuals = (y_val.to_numpy() - val_preds)
        
        # Estimate reference uncertainty
        self.reference_uncertainties = self.estimate_uncertainty(X_val)

        logger.info("Baseline validation performance -> RMSE: %.4f, MAE: %.4f, R2: %.4f", val_rmse, val_mae, val_r2)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Standard batch inference."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        X = df[self.features] if isinstance(df, pd.DataFrame) else df
        return self.model.predict(X)

    def estimate_uncertainty(self, df: pd.DataFrame) -> np.ndarray:
        """
        Estimates epistemic prediction uncertainty via tree ensemble leaf variance.
        
        Statistical principle:
        For an input vector x, the booster contains T individual decision trees.
        When x lands in familiar reference space, all trees agree closely on leaf predictions.
        When x shifts into out-of-distribution / drifted space, trees extrapolate diverge,
        producing a spike in ensemble standard deviation:
            sigma_tree(x) = sqrt( 1/T * sum_{t=1}^T (f_t(x) - f_bar(x))^2 )
        """
        if self.model is None or self.model.booster_ is None:
            raise RuntimeError("Model booster not initialized.")

        X = df[self.features] if isinstance(df, pd.DataFrame) else df
        booster = self.model.booster_

        # Fast calculation of tree individual leaf predictions across iterations
        # LightGBM booster.predict with raw_score or per-tree predictions
        leaf_indices = booster.predict(X, pred_leaf=True)
        # Compute leaf variance across trees
        tree_preds = []
        n_trees = booster.num_trees()
        step = max(1, n_trees // 20)  # sample 20 trees for fast inference
        for t in range(0, n_trees, step):
            p_t = booster.predict(X, num_iteration=t+1)
            tree_preds.append(p_t)

        tree_matrix = np.array(tree_preds)  # shape (n_sample_trees, n_rows)
        uncertainty_std = np.std(tree_matrix, axis=0)
        return uncertainty_std

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluates model performance if ground truth labels are available."""
        if self.target_col not in df.columns:
            raise ValueError(f"Ground truth target '{self.target_col}' not found in dataframe.")
        y_true = df[self.target_col]
        y_pred = self.predict(df)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "mean_pred": float(np.mean(y_pred)),
            "std_pred": float(np.std(y_pred)),
        }

    def save(self, filepath: str = "./models/baseline_lgb_model.joblib") -> None:
        """Serializes model and metadata to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "features": self.features,
            "target_col": self.target_col,
            "params": self.params,
            "reference_metrics": self.reference_metrics,
            "reference_uncertainties": self.reference_uncertainties,
        }
        joblib.dump(payload, path)
        logger.info("Saved baseline model artifact to %s", filepath)

    @classmethod
    def load(cls, filepath: str = "./models/baseline_lgb_model.joblib") -> "BaselineTaxiModel":
        """Loads serialized model artifact."""
        payload = joblib.load(filepath)
        instance = cls(features=payload["features"], target_col=payload["target_col"])
        instance.model = payload["model"]
        instance.params = payload["params"]
        instance.reference_metrics = payload["reference_metrics"]
        instance.reference_uncertainties = payload.get("reference_uncertainties")
        return instance


def train_baseline_model(
    reference_df: pd.DataFrame,
    model_save_path: str = "./models/baseline_lgb_model.joblib",
) -> BaselineTaxiModel:
    """Convenience function to train and save the baseline model on reference data."""
    model = BaselineTaxiModel()
    model.fit(reference_df)
    model.save(model_save_path)
    return model

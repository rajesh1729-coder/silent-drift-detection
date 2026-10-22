"""
Baseline Model Training and Inference with Uncertainty Estimation.
Uses Gradient Boosted Trees with staged prediction variance estimation.
"""

import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .preprocess import ALL_MODEL_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)


class BaselineTaxiModel:
    """
    Production Gradient Boosted Regressor for predicting NYC Taxi Tip Amounts ($).
    Equipped with staged tree-ensemble variance estimators for silent/uncertainty drift detection.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        target_col: str = TARGET_COLUMN,
        n_estimators: int = 80,
        learning_rate: float = 0.08,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        self.features = features or ALL_MODEL_FEATURES
        self.target_col = target_col
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": random_state,
        }
        self.model: Optional[GradientBoostingRegressor] = None
        self.reference_metrics: Dict[str, float] = {}
        self.reference_predictions: Optional[np.ndarray] = None
        self.reference_residuals: Optional[np.ndarray] = None
        self.reference_uncertainties: Optional[np.ndarray] = None

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "BaselineTaxiModel":
        """Trains Gradient Boosted model on reference features and records baseline metrics."""
        X_train = train_df[self.features]
        y_train = train_df[self.target_col]

        if val_df is None:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.params["random_state"]
            )
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = val_df[self.features], val_df[self.target_col]

        logger.info("Training Gradient Boosting model on %d samples with %d features...", len(X_tr), len(self.features))
        self.model = GradientBoostingRegressor(**self.params)
        self.model.fit(X_tr, y_tr)

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
        self.reference_residuals = y_val.to_numpy() - val_preds
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
        Estimates epistemic prediction uncertainty via staged tree ensemble dispersion.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        X = df[self.features] if isinstance(df, pd.DataFrame) else df
        step = max(1, self.model.n_estimators_ // 15)
        staged_preds = [
            p for idx, p in enumerate(self.model.staged_predict(X)) if idx % step == 0
        ]

        if not staged_preds:
            staged_preds.append(self.model.predict(X))

        staged_matrix = np.array(staged_preds)
        return np.std(staged_matrix, axis=0)

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

    def save(self, filepath: str = "./models/baseline_gb_model.joblib") -> None:
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
    def load(cls, filepath: str = "./models/baseline_gb_model.joblib") -> "BaselineTaxiModel":
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
    model_save_path: str = "./models/baseline_gb_model.joblib",
) -> BaselineTaxiModel:
    """Convenience function to train and save the baseline model on reference data."""
    model = BaselineTaxiModel()
    model.fit(reference_df)
    model.save(model_save_path)
    return model
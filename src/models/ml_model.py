"""Machine-learning model builders.

All models are evaluated with a *walk-forward* protocol (see
``src/models/walkforward.py``): rolling re-training on a purged window,
never shuffled. Linear models are wrapped in a scaling pipeline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split (X, y) chronologically into train and test sets.

    Kept as a simple alternative to walk-forward for quick experiments.
    Returns ``(X_train, X_test, y_train, y_test)``.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    split_index = int(len(X) * train_ratio)
    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:],
    )


def build_ridge(params: dict[str, Any]) -> Pipeline:
    """Standardized Ridge regression."""
    return make_pipeline(StandardScaler(), Ridge(**params))


def build_lasso(params: dict[str, Any]) -> Pipeline:
    """Standardized Lasso regression."""
    return make_pipeline(StandardScaler(), Lasso(max_iter=50_000, **params))


def build_random_forest(params: dict[str, Any]) -> RandomForestRegressor:
    """Instantiate a RandomForestRegressor from config parameters."""
    return RandomForestRegressor(**params)


def build_gradient_boosting(params: dict[str, Any]) -> GradientBoostingRegressor:
    """Instantiate a GradientBoostingRegressor from config parameters."""
    return GradientBoostingRegressor(**params)


def build_quantile_gradient_boosting(
    params: dict[str, Any],
    quantile: float,
) -> GradientBoostingRegressor:
    """Gradient boosting predicting a conditional quantile of the target."""
    return GradientBoostingRegressor(loss="quantile", alpha=quantile, **params)


def feature_importances(model: Any, feature_names: list[str]) -> pd.Series:
    """Return sorted feature importances for tree-based models."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)

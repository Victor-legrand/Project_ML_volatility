"""Machine-learning model builders.

All models are evaluated with a *walk-forward* protocol (see
``src/models/walkforward.py``): rolling re-training on a purged window,
never shuffled. Linear models are wrapped in a scaling pipeline.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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


def build_extra_trees(params: dict[str, Any]) -> ExtraTreesRegressor:
    """Extremely randomized trees: lower variance than RF via random splits."""
    return ExtraTreesRegressor(**params)


class TimeSeriesTunedModel:
    """Model whose hyperparameters are re-tuned on every ``fit`` call.

    Runs a randomized search over a *temporal* cross-validation
    (``TimeSeriesSplit``: each fold trains on the past and validates on
    the future) inside the training window it receives. Plugged into the
    walk-forward protocol, this is nested temporal CV: hyperparameters
    adapt to each regime without ever seeing the prediction block.
    """

    def __init__(
        self,
        base_factory: Any,
        param_grid: dict[str, list[Any]],
        n_iter: int = 15,
        cv_splits: int = 3,
        random_state: int = 42,
    ) -> None:
        self._base_factory = base_factory
        self._param_grid = param_grid
        self._n_iter = n_iter
        self._cv_splits = cv_splits
        self._random_state = random_state
        self.best_params_: dict[str, Any] | None = None
        self._best_model: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TimeSeriesTunedModel":
        search = RandomizedSearchCV(
            estimator=self._base_factory(),
            param_distributions=self._param_grid,
            n_iter=self._n_iter,
            cv=TimeSeriesSplit(n_splits=self._cv_splits),
            scoring="neg_mean_squared_error",
            random_state=self._random_state,
            n_jobs=-1,
        )
        search.fit(X, y)
        self.best_params_ = search.best_params_
        self._best_model = search.best_estimator_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._best_model is None:
            raise RuntimeError("TimeSeriesTunedModel must be fitted before predict.")
        return self._best_model.predict(X)


def build_tuned_hist_gradient_boosting(config: dict[str, Any]) -> TimeSeriesTunedModel:
    """HistGradientBoosting re-tuned by temporal CV in each window."""
    random_state = config["random_state"]
    # functools.partial (not a lambda) keeps the fitted model picklable.
    return TimeSeriesTunedModel(
        base_factory=partial(HistGradientBoostingRegressor, random_state=random_state),
        param_grid=config["param_grid"],
        n_iter=config["tuning_iterations"],
        cv_splits=config["cv_splits"],
        random_state=random_state,
    )


def build_stacking_meta_model(alpha: float = 1.0) -> Ridge:
    """Non-negative Ridge combining base-model out-of-sample predictions.

    Non-negativity keeps the combination interpretable (weights are
    contributions, not spread trades between correlated forecasts) and
    guards against overfitting the meta level.
    """
    return Ridge(alpha=alpha, positive=True)


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

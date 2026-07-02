"""HAR-RV baseline models (Corsi, 2009).

The Heterogeneous AutoRegressive model of Realized Volatility regresses
future realized volatility on realized volatilities computed over daily,
weekly and monthly horizons. With daily data only, we proxy those three
components with rolling realized vols (e.g. 5, 20 and 60 days), which
keeps the spirit of the model: volatility cascades across horizons.

``HARXModel`` adds the implied volatility level as an exogenous
regressor — the proper academic benchmark when option data is available.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


class HARModel:
    """Linear HAR-RV model on multi-horizon realized volatilities."""

    def __init__(
        self,
        vol_windows: list[int],
        extra_features: list[str] | None = None,
    ) -> None:
        self.feature_names: list[str] = [f"rv_{w}" for w in sorted(vol_windows)]
        if extra_features:
            self.feature_names += list(extra_features)
        self._regression = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARModel":
        """Fit the linear regression on the HAR components only."""
        self._regression.fit(X[self.feature_names], y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the target for each row of X."""
        predictions = self._regression.predict(X[self.feature_names])
        return pd.Series(predictions, index=X.index)

    @property
    def coefficients(self) -> pd.Series:
        """Fitted coefficients, indexed by feature name."""
        return pd.Series(self._regression.coef_, index=self.feature_names)


class HARXModel(HARModel):
    """HAR-RV augmented with implied volatility (HAR-X)."""

    def __init__(self, vol_windows: list[int]) -> None:
        super().__init__(vol_windows, extra_features=["implied_vol"])

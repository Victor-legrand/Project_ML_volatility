"""Realized volatility features, target construction and transforms.

Look-ahead policy
-----------------
Every feature at date ``t`` only uses information available at the close
of ``t`` (rolling windows ending at ``t``). The target at date ``t`` is
built from the realized volatility over the *next* ``horizon`` trading
days (``t+1`` .. ``t+horizon``), so a model trained on
(features_t, target_t) is a genuine forecast.

Target transforms
-----------------
``rv``            future RV level (annualized).
``log_rv``        log of future RV — stabilizes the right-skewed
                  distribution of volatility.
``log_rv_ratio``  log(future RV / implied vol) — the realized
                  variance-risk-premium residual: what the option market
                  did *not* already price. This is the quantity the
                  strategy actually trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "target"
FUTURE_RV_COLUMN = "future_rv"
NON_FEATURE_COLUMNS = [TARGET_COLUMN, FUTURE_RV_COLUMN]

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Volatility estimators
# ---------------------------------------------------------------------------

def realized_volatility(
    returns: pd.Series,
    window: int,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Annualized close-to-close realized volatility (rolling std)."""
    return returns.rolling(window).std() * np.sqrt(trading_days_per_year)


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Annualized Parkinson (1980) high-low volatility estimator.

    sigma^2 = (1 / 4 ln 2) * E[ln(H/L)^2] — about 5x more statistically
    efficient than the close-to-close estimator.
    """
    hl_squared = np.log(high / low) ** 2 / (4.0 * np.log(2.0))
    return np.sqrt(hl_squared.rolling(window).mean() * trading_days_per_year)


def garman_klass_volatility(
    ohlc: pd.DataFrame,
    window: int,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Annualized Garman-Klass (1980) OHLC volatility estimator.

    sigma^2 = 0.5 * ln(H/L)^2 - (2 ln 2 - 1) * ln(C/O)^2. Uses the full
    intraday range, more efficient than close-to-close.
    """
    log_hl = np.log(ohlc["High"] / ohlc["Low"])
    log_co = np.log(ohlc["Close"] / ohlc["Open"])
    variance = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    variance = variance.clip(lower=0.0)
    return np.sqrt(variance.rolling(window).mean() * trading_days_per_year)


def negative_semivolatility(
    returns: pd.Series,
    window: int,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Annualized downside semi-volatility (leverage effect, LHAR-style).

    Uses only negative returns: sqrt((252 / window) * sum r^2 1{r<0}).
    """
    downside = returns.where(returns < 0, 0.0) ** 2
    return np.sqrt(
        downside.rolling(window).sum() * trading_days_per_year / window
    )


# ---------------------------------------------------------------------------
# Path-dependent features
# ---------------------------------------------------------------------------

def days_since_shock(
    returns: pd.Series,
    sigma_window: int = 60,
    threshold: float = 2.0,
    cap: int = 252,
) -> pd.Series:
    """Number of days since the last |return| > threshold * rolling sigma.

    The sigma used at date ``t`` is computed up to ``t-1`` (no look-ahead).
    Capped at ``cap`` days; days before the first shock get the cap.
    """
    sigma = returns.rolling(sigma_window).std().shift(1)
    is_shock = returns.abs() > threshold * sigma
    positions = np.arange(len(returns), dtype=float)
    last_shock = pd.Series(
        np.where(is_shock, positions, np.nan), index=returns.index
    ).ffill()
    elapsed = pd.Series(positions, index=returns.index) - last_shock
    return elapsed.fillna(float(cap)).clip(upper=cap)


def drawdown_from_peak(prices: pd.Series) -> pd.Series:
    """Current drawdown from the running maximum (negative fraction)."""
    return prices / prices.cummax() - 1.0


# ---------------------------------------------------------------------------
# Target transforms
# ---------------------------------------------------------------------------

def future_realized_volatility(
    returns: pd.Series,
    horizon: int,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Annualized realized volatility over the next ``horizon`` days.

    At date ``t`` this uses returns from ``t+1`` to ``t+horizon``.
    """
    rv = realized_volatility(returns, horizon, trading_days_per_year)
    return rv.shift(-horizon)


def transform_target(
    future_rv: pd.Series,
    implied_vol: pd.Series,
    target_type: str,
) -> pd.Series:
    """Map future RV to the model target space (see module docstring)."""
    if target_type == "rv":
        return future_rv
    if target_type == "log_rv":
        return np.log(future_rv.clip(lower=_EPS))
    if target_type == "log_rv_ratio":
        return np.log(future_rv.clip(lower=_EPS) / implied_vol.clip(lower=_EPS))
    raise ValueError(f"Unknown target_type: {target_type!r}")


def invert_target(
    predictions: pd.Series,
    implied_vol: pd.Series,
    target_type: str,
) -> pd.Series:
    """Map predictions back from target space to an RV level (annualized)."""
    if target_type == "rv":
        return predictions
    if target_type == "log_rv":
        return np.exp(predictions)
    if target_type == "log_rv_ratio":
        return np.exp(predictions) * implied_vol.reindex(predictions.index)
    raise ValueError(f"Unknown target_type: {target_type!r}")


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    target_ohlc: pd.DataFrame,
    target_ticker: str,
    implied_vol_index: pd.Series,
    implied_vol_3m_index: pd.Series,
    vol_windows: list[int],
    gk_windows: list[int],
    target_horizon: int,
    target_type: str = "log_rv_ratio",
    shock_sigma_window: int = 60,
    shock_threshold: float = 2.0,
    trading_days_per_year: int = 252,
    require_target: bool = True,
) -> pd.DataFrame:
    """Build the modelling dataset for one target ticker.

    Returns a dataframe with feature columns plus ``target`` (transformed
    prediction target) and ``future_rv`` (raw future RV level, kept for
    evaluation only — never feed it to a model). Rows with incomplete
    features are dropped; with ``require_target=False`` the most recent
    rows (whose future window is not observed yet) are kept with a NaN
    target — needed for live inference on today's data.
    """
    target_returns = returns[target_ticker]
    features = pd.DataFrame(index=returns.index)
    days_per_year = trading_days_per_year

    # Close-to-close realized vols and lags.
    for window in vol_windows:
        features[f"rv_{window}"] = realized_volatility(
            target_returns, window, days_per_year
        )
    shortest = min(vol_windows)
    features[f"rv_{shortest}_lag1"] = features[f"rv_{shortest}"].shift(1)
    features[f"rv_{shortest}_lag5"] = features[f"rv_{shortest}"].shift(5)

    # Range-based estimators (more efficient than close-to-close).
    ohlc = target_ohlc.reindex(features.index)
    for window in gk_windows:
        features[f"gk_{window}"] = garman_klass_volatility(ohlc, window, days_per_year)
    features["park_20"] = parkinson_volatility(
        ohlc["High"], ohlc["Low"], 20, days_per_year
    )

    # Leverage effect: downside semi-volatility.
    features["semi_rv_20"] = negative_semivolatility(target_returns, 20, days_per_year)

    # Cross-asset realized vols (shortest window).
    for ticker in returns.columns:
        if ticker != target_ticker:
            safe_name = ticker.replace("^", "").lower()
            features[f"rv_{shortest}_{safe_name}"] = realized_volatility(
                returns[ticker], shortest, days_per_year
            )

    # Implied volatility level, vol risk premium proxy, term structure.
    reference_window = 20 if 20 in vol_windows else shortest
    implied = implied_vol_index.reindex(features.index) / 100.0
    implied_3m = implied_vol_3m_index.reindex(features.index) / 100.0
    features["implied_vol"] = implied
    features["implied_minus_rv"] = implied - features[f"rv_{reference_window}"]
    features["vix_term_structure"] = implied / implied_3m

    # Returns, shock memory, drawdown, day-of-week seasonality.
    features["return_1d"] = target_returns
    features["return_5d"] = target_returns.rolling(5).sum()
    features["days_since_shock"] = days_since_shock(
        target_returns, shock_sigma_window, shock_threshold
    )
    features["drawdown"] = drawdown_from_peak(prices[target_ticker])
    day_of_week = features.index.dayofweek
    for day in range(1, 5):  # Monday is the baseline
        features[f"dow_{day}"] = (day_of_week == day).astype(float)

    # Target (transformed) + raw future RV kept for evaluation.
    future_rv = future_realized_volatility(target_returns, target_horizon, days_per_year)
    features[FUTURE_RV_COLUMN] = future_rv
    features[TARGET_COLUMN] = transform_target(future_rv, implied, target_type)

    feature_columns = [c for c in features.columns if c not in NON_FEATURE_COLUMNS]
    features = features.dropna(subset=feature_columns)
    if require_target:
        features = features.dropna(subset=NON_FEATURE_COLUMNS)
    return features


def split_features_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split the modelling dataset into (X, y, future_rv).

    ``future_rv`` is the raw future RV level, used only for evaluation.
    """
    X = dataset.drop(columns=NON_FEATURE_COLUMNS)
    return X, dataset[TARGET_COLUMN], dataset[FUTURE_RV_COLUMN]

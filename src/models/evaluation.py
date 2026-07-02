"""Forecast evaluation: error metrics, sub-period stability, DM test.

All comparisons are done in RV space (annualized volatility level) so
that models trained on different target transforms — and naive
benchmarks — are directly comparable.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute RMSE, MAE and R² for a single prediction series."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluation_table(y_true: pd.Series, predictions: pd.DataFrame) -> pd.DataFrame:
    """RMSE / MAE / R² for every prediction column, sorted by RMSE."""
    rows = {
        name: evaluate_predictions(y_true, predictions[name])
        for name in predictions.columns
    }
    return pd.DataFrame(rows).T.sort_values("rmse")


def rmse_by_period(
    y_true: pd.Series,
    predictions: pd.DataFrame,
    freq: str = "YE",
) -> pd.DataFrame:
    """RMSE per calendar period (year by default), one row per period.

    A model whose global RMSE hides a failure outside crisis periods
    shows up immediately here.
    """
    squared_errors = (predictions.sub(y_true, axis=0)) ** 2
    per_period = squared_errors.resample(freq).mean() ** 0.5
    per_period.index = per_period.index.year
    per_period.index.name = "year"
    return per_period


def diebold_mariano(
    errors_a: pd.Series,
    errors_b: pd.Series,
    horizon: int = 5,
) -> tuple[float, float]:
    """Diebold-Mariano test on squared-error loss.

    Tests H0: equal predictive accuracy of forecasts A and B. The loss
    differential d_t = e_a^2 - e_b^2 is autocorrelated for multi-step
    forecasts, so its long-run variance uses Bartlett weights up to
    ``horizon - 1`` lags (Newey-West).

    Returns
    -------
    (statistic, p_value):
        Negative statistic means A is more accurate than B. Two-sided
        p-value from the normal approximation.
    """
    a, b = errors_a.align(errors_b, join="inner")
    d = (a**2 - b**2).to_numpy()
    n = len(d)
    if n < 10:
        raise ValueError(f"Too few observations for the DM test: {n}")
    d_mean = d.mean()
    d_centered = d - d_mean

    max_lag = max(horizon - 1, 0)
    long_run_var = float(d_centered @ d_centered) / n
    for lag in range(1, max_lag + 1):
        autocov = float(d_centered[lag:] @ d_centered[:-lag]) / n
        long_run_var += 2.0 * (1.0 - lag / (max_lag + 1)) * autocov
    if long_run_var <= 0:
        return 0.0, 1.0

    statistic = d_mean / math.sqrt(long_run_var / n)
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return float(statistic), float(p_value)


def dm_table(
    y_true: pd.Series,
    predictions: pd.DataFrame,
    benchmark: str,
    horizon: int = 5,
) -> pd.DataFrame:
    """DM test of every model against a benchmark column.

    ``dm_stat < 0`` means the model beats the benchmark.
    """
    benchmark_errors = predictions[benchmark] - y_true
    rows = {}
    for name in predictions.columns:
        if name == benchmark:
            continue
        stat, p_value = diebold_mariano(
            predictions[name] - y_true, benchmark_errors, horizon
        )
        rows[name] = {"dm_stat": stat, "p_value": p_value}
    return pd.DataFrame(rows).T.sort_values("dm_stat")

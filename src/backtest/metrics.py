"""Financial performance metrics for daily return series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio of a daily return series.

    ``risk_free_rate`` is annualized and converted to a daily rate.
    Returns 0.0 when volatility is zero.
    """
    daily_rf = risk_free_rate / trading_days_per_year
    excess = returns - daily_rf
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(excess.mean() / vol * np.sqrt(trading_days_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.25 for -25%)."""
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1.0
    return float(drawdowns.min())


def cagr(equity_curve: pd.Series, trading_days_per_year: int = 252) -> float:
    """Compound annual growth rate of an equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if total_return <= 0:
        return -1.0
    years = len(equity_curve) / trading_days_per_year
    return float(total_return ** (1.0 / years) - 1.0)


def annualized_volatility(
    returns: pd.Series,
    trading_days_per_year: int = 252,
) -> float:
    """Annualized volatility of a daily return series."""
    return float(returns.std() * np.sqrt(trading_days_per_year))


def performance_summary(
    net_returns: pd.Series,
    equity_curve: pd.Series,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    """Compute the standard performance report for a backtest."""
    return {
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0),
        "cagr": cagr(equity_curve, trading_days_per_year),
        "sharpe_ratio": sharpe_ratio(net_returns, trading_days_per_year=trading_days_per_year),
        "annualized_volatility": annualized_volatility(net_returns, trading_days_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "n_days": float(len(net_returns)),
    }

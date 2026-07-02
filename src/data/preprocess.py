"""Cleaning of raw prices and computation of log returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_prices(
    prices: pd.DataFrame,
    required_columns: list[str] | None = None,
    max_ffill_days: int = 3,
) -> pd.DataFrame:
    """Clean a wide dataframe of daily close prices.

    Steps: drop rows where all tickers are missing, forward-fill small
    gaps (holidays that differ across instruments), then drop rows where
    a *required* ticker is still missing. Non-required tickers (e.g.
    tradable proxies with a later inception date) may keep leading NaNs.

    Parameters
    ----------
    prices:
        Wide dataframe of close prices, one column per ticker.
    required_columns:
        Tickers that must be present on every row. Defaults to all.
    max_ffill_days:
        Maximum number of consecutive days to forward-fill.
    """
    required = list(required_columns) if required_columns else list(prices.columns)
    cleaned = prices.dropna(how="all")
    cleaned = cleaned.ffill(limit=max_ffill_days)
    cleaned = cleaned.dropna(subset=required)
    if (cleaned <= 0).any().any():
        raise ValueError("Non-positive prices found after cleaning.")
    return cleaned


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from close prices.

    Columns with a later inception keep NaN before their first two
    observations; rows where every column is NaN are dropped.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how="all")

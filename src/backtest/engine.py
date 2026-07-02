"""Vectorized daily backtest engine with two-leg execution.

A net volatility position in [-1, 1] is expressed with two long-only
legs: a long-vol proxy (e.g. VIXY) when the position is positive, and a
short-vol proxy (an inverse ETP such as SVXY) when negative — shorting
ETPs outright is impractical for a research backtest.

Anti look-ahead convention
--------------------------
Weights decided at the close of day ``t`` earn the legs' returns of day
``t+1``: the engine shifts weights by one day before multiplying by
returns. Transaction costs are charged on every change of leg weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

LEG_COLUMNS = ["long", "short"]


@dataclass
class BacktestResult:
    """Container for backtest outputs (all series share the same index)."""

    positions: pd.Series          # net vol exposure actually held
    gross_returns: pd.Series
    costs: pd.Series
    net_returns: pd.Series
    equity_curve: pd.Series


def position_to_leg_weights(position: pd.Series) -> pd.DataFrame:
    """Map a net position in [-1, 1] to long-only weights on each leg."""
    return pd.DataFrame(
        {"long": position.clip(lower=0.0), "short": (-position).clip(lower=0.0)}
    )


def run_backtest(
    weights: pd.DataFrame,
    leg_returns: pd.DataFrame,
    transaction_cost_bps: float,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Backtest leg weights against leg returns.

    Parameters
    ----------
    weights:
        Long-only weights per leg (columns ``long`` and ``short``),
        decided at the close of each date.
    leg_returns:
        Daily returns of each leg, same columns.
    transaction_cost_bps:
        One-way cost in basis points, applied to each unit of weight
        change on each leg.
    initial_capital:
        Starting equity used to scale the equity curve.
    """
    common_index = weights.index.intersection(leg_returns.dropna().index)
    aligned_weights = weights.loc[common_index, LEG_COLUMNS]
    aligned_returns = leg_returns.loc[common_index, LEG_COLUMNS]

    # Weights held during day t were decided at the close of t-1.
    held = aligned_weights.shift(1).fillna(0.0)
    gross_returns = (held * aligned_returns).sum(axis=1)

    cost_rate = transaction_cost_bps / 10_000.0
    weight_changes = held.diff().abs()
    weight_changes.iloc[0] = held.iloc[0].abs()
    costs = weight_changes.sum(axis=1) * cost_rate

    net_returns = gross_returns - costs
    equity_curve = initial_capital * (1.0 + net_returns).cumprod()
    net_position = held["long"] - held["short"]

    return BacktestResult(
        positions=net_position.rename("position"),
        gross_returns=gross_returns.rename("gross_return"),
        costs=costs.rename("cost"),
        net_returns=net_returns.rename("net_return"),
        equity_curve=equity_curve.rename("equity"),
    )

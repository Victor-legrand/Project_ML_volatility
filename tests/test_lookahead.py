"""Anti look-ahead regression tests.

These tests guard the three places where future information could leak:
the target construction, the backtest execution lag, and the
walk-forward purge gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import position_to_leg_weights, run_backtest
from src.features.volatility_features import future_realized_volatility
from src.models.walkforward import walk_forward_predictions


def _daily_index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def test_target_only_uses_future_returns() -> None:
    """The target at date t must equal the RV of returns t+1 .. t+h."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 100), index=_daily_index(100))
    horizon = 5

    target = future_realized_volatility(returns, horizon, trading_days_per_year=252)

    t = 30
    future_window = returns.iloc[t + 1 : t + 1 + horizon]
    expected = future_window.std() * np.sqrt(252)
    assert target.iloc[t] == pytest.approx(expected)
    # The last `horizon` dates have no complete future window.
    assert target.iloc[-horizon:].isna().all()


def test_backtest_position_lagged_one_day() -> None:
    """A weight decided at close of day t must earn the return of t+1."""
    index = _daily_index(3)
    big_return = 0.10
    leg_returns = pd.DataFrame(
        {"long": [0.0, big_return, 0.0], "short": [0.0, 0.0, 0.0]}, index=index
    )
    # The signal "knows" the big return of day index[1] and goes long
    # that same day: it must NOT capture it (execution at the close).
    position = pd.Series([0.0, 1.0, 0.0], index=index)

    result = run_backtest(
        position_to_leg_weights(position), leg_returns, transaction_cost_bps=0.0
    )

    # Day index[1]'s return is earned by the position decided at the
    # close of index[0], which was flat.
    assert result.net_returns.iloc[1] == pytest.approx(0.0)
    # The long decided at close of index[1] earns index[2]'s return (0).
    assert result.net_returns.iloc[2] == pytest.approx(0.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(100_000.0)


def test_backtest_costs_charged_on_changes() -> None:
    """Costs apply to each unit of weight change, per leg."""
    index = _daily_index(4)
    leg_returns = pd.DataFrame(0.0, index=index, columns=["long", "short"])
    position = pd.Series([1.0, -1.0, -1.0, -1.0], index=index)

    result = run_backtest(
        position_to_leg_weights(position), leg_returns, transaction_cost_bps=100.0
    )

    cost_rate = 100.0 / 10_000.0
    # Held weights (positions shifted by one day):
    # index[1]: long leg 0 -> 1 (1 unit traded);
    # index[2]: long 1 -> 0 and short 0 -> 1 (2 units traded).
    assert result.costs.iloc[1] == pytest.approx(1.0 * cost_rate)
    assert result.costs.iloc[2] == pytest.approx(2.0 * cost_rate)
    assert result.costs.iloc[3] == pytest.approx(0.0)


class _IndexRecorder:
    """Fake model recording the training indices it was fitted on."""

    def __init__(self, log: list[pd.Index]) -> None:
        self._log = log

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_IndexRecorder":
        self._log.append(X.index)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


def test_walkforward_purge_gap() -> None:
    """Training data must end at least `purge` rows before predictions."""
    n, train_window, refit_every, purge = 300, 100, 20, 5
    index = _daily_index(n)
    X = pd.DataFrame({"x": np.arange(n, dtype=float)}, index=index)
    y = pd.Series(np.arange(n, dtype=float), index=index)
    train_logs: list[pd.Index] = []

    predictions, _ = walk_forward_predictions(
        X, y, {"recorder": lambda: _IndexRecorder(train_logs)},
        train_window=train_window, refit_every=refit_every, purge=purge,
    )

    positions = pd.Series(np.arange(n), index=index)
    first_predicted = int(positions[predictions.index[0]])
    assert first_predicted == train_window + purge
    for block_number, train_index in enumerate(train_logs):
        block_start = train_window + purge + block_number * refit_every
        last_train_row = int(positions[train_index[-1]])
        assert last_train_row <= block_start - purge - 1
        assert len(train_index) <= train_window

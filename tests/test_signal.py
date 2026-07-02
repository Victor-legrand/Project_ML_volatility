"""Tests for the signal construction (hysteresis, sizing, vol targeting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.volatility_signal import (
    conviction_size,
    hysteresis_direction,
    vol_target_leverage,
    volatility_score,
)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_score_is_log_ratio() -> None:
    predicted = _series([0.20, 0.10])
    implied = _series([0.10, 0.20])
    score = volatility_score(predicted, implied)
    assert score.iloc[0] == pytest.approx(np.log(2.0))
    assert score.iloc[1] == pytest.approx(-np.log(2.0))


def test_hysteresis_holds_position_inside_band() -> None:
    # Crosses entry (0.10), then hovers between exit (0.03) and entry:
    # the position must be held, not flip-flopped.
    score = _series([0.05, 0.12, 0.06, 0.04, 0.02, 0.05])
    direction = hysteresis_direction(score, entry_threshold=0.10, exit_threshold=0.03)
    assert direction.tolist() == [0, 1, 1, 1, 0, 0]


def test_hysteresis_flips_through_opposite_entry() -> None:
    score = _series([0.15, -0.02, -0.15, -0.05])
    direction = hysteresis_direction(score, entry_threshold=0.10, exit_threshold=0.03)
    assert direction.tolist() == [1, 0, -1, -1]


def test_conviction_size_bounded_and_quantile_halving() -> None:
    score = _series([0.15, 0.60, -0.15])
    direction = _series([1.0, 1.0, -1.0]).astype(int)
    q10 = _series([-0.05, 0.10, -0.30])   # straddles 0 on day 1 only
    q90 = _series([0.40, 0.90, -0.05])

    position = conviction_size(score, direction, sizing_scale=0.30,
                               score_q10=q10, score_q90=q90)

    assert position.iloc[0] == pytest.approx(0.5 * 0.15 / 0.30)  # halved
    assert position.iloc[1] == pytest.approx(1.0)                # capped
    assert position.iloc[2] == pytest.approx(-0.15 / 0.30)       # confident short


def test_vol_target_reduces_leverage_when_proxy_is_wild() -> None:
    n = 60
    index = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(1)
    quiet = pd.Series(rng.normal(0, 0.001, n), index=index)   # ~1.6% ann.
    wild = pd.Series(rng.normal(0, 0.05, n), index=index)     # ~80% ann.
    leg_returns = pd.DataFrame({"long": wild, "short": quiet})
    position = pd.Series(1.0, index=index)

    scaled = vol_target_leverage(position, leg_returns, vol_target=0.30,
                                 vol_window=20, max_leverage=1.0)

    assert (scaled.iloc[25:] < 1.0).all()      # de-levered on the wild leg
    assert (scaled <= 1.0).all() and (scaled >= 0.0).all()

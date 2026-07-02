"""Volatility trading signal: score, hysteresis, sizing, vol targeting.

Pipeline
--------
1. **Score**: ``log(predicted RV / implied vol)``. Positive means the
   model expects more volatility than the option market prices (vol is
   cheap -> long vol); negative means vol is expensive -> short vol.
2. **Hysteresis**: enter a position only when |score| > entry_threshold,
   exit only when |score| < exit_threshold. The asymmetric band kills
   the flip-flopping (and transaction costs) of a single threshold.
3. **Sizing**: position magnitude proportional to conviction,
   ``min(|score| / sizing_scale, 1)``. Optionally halved when the
   quantile band [q10, q90] straddles zero (uncertain sign).
4. **Vol targeting**: the position is scaled by
   ``vol_target / realized vol of the traded leg`` (capped) so the
   strategy risk does not explode with the proxy's own volatility.

The signal at date ``t`` only uses information available at ``t``; the
backtest engine applies it to next-day returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_score(predicted_rv: pd.Series, implied_vol: pd.Series) -> pd.Series:
    """Log-ratio of predicted realized vol to implied vol."""
    aligned_pred, aligned_iv = predicted_rv.align(implied_vol, join="inner")
    return np.log(aligned_pred / aligned_iv).rename("score")


def hysteresis_direction(
    score: pd.Series,
    entry_threshold: float,
    exit_threshold: float,
) -> pd.Series:
    """Convert a score into a {-1, 0, +1} direction with hysteresis.

    A position opens when |score| crosses ``entry_threshold`` and closes
    only when |score| falls below ``exit_threshold`` (or flips if the
    score crosses the opposite entry level).
    """
    if not 0 <= exit_threshold <= entry_threshold:
        raise ValueError(
            f"Need 0 <= exit ({exit_threshold}) <= entry ({entry_threshold})"
        )
    directions = np.zeros(len(score), dtype=int)
    state = 0
    for i, value in enumerate(score.to_numpy()):
        if state == 1 and value < exit_threshold:
            state = 0
        elif state == -1 and value > -exit_threshold:
            state = 0
        if state == 0:
            if value > entry_threshold:
                state = 1
            elif value < -entry_threshold:
                state = -1
        directions[i] = state
    return pd.Series(directions, index=score.index, name="direction")


def conviction_size(
    score: pd.Series,
    direction: pd.Series,
    sizing_scale: float,
    score_q10: pd.Series | None = None,
    score_q90: pd.Series | None = None,
) -> pd.Series:
    """Position in [-1, 1]: direction times conviction-based magnitude.

    If quantile scores are provided and the [q10, q90] band straddles
    zero (the sign of the edge is uncertain), the magnitude is halved.
    """
    if sizing_scale <= 0:
        raise ValueError(f"sizing_scale must be > 0, got {sizing_scale}")
    magnitude = (score.abs() / sizing_scale).clip(upper=1.0)
    if score_q10 is not None and score_q90 is not None:
        uncertain_sign = (score_q10.reindex(score.index) < 0) & (
            score_q90.reindex(score.index) > 0
        )
        magnitude = magnitude.where(~uncertain_sign, magnitude * 0.5)
    return (direction * magnitude).rename("position")


def vol_target_leverage(
    position: pd.Series,
    leg_returns: pd.DataFrame,
    vol_target: float,
    vol_window: int = 20,
    max_leverage: float = 1.0,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Scale positions so the traded leg runs near ``vol_target``.

    ``leg_returns`` must have columns ``long`` and ``short``: the
    long-vol proxy vol is used when position > 0, the short-vol proxy
    vol when position < 0. Vols are rolling and only use past data.
    """
    leg_vols = leg_returns.rolling(vol_window).std() * np.sqrt(trading_days_per_year)
    active_vol = leg_vols["long"].where(position >= 0, leg_vols["short"])
    active_vol = active_vol.reindex(position.index)
    leverage = (vol_target / active_vol).clip(upper=max_leverage)
    return (position * leverage).fillna(0.0).rename("position")


def signal_statistics(position: pd.Series) -> dict[str, float]:
    """Diagnostics: exposure shares, average size and daily turnover."""
    total = len(position)
    changes = position.diff().abs().fillna(0)
    return {
        "long_share": float((position > 0).sum() / total),
        "short_share": float((position < 0).sum() / total),
        "flat_share": float((position == 0).sum() / total),
        "avg_abs_position": float(position.abs().mean()),
        "avg_daily_turnover": float(changes.mean()),
    }

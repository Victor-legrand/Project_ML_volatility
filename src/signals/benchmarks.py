"""Model-free benchmark strategies.

These are the opponents the ML strategy must beat *in P&L terms* to
justify its existence. They receive exactly the same vol targeting,
transaction costs and backtest engine as the ML strategy — only the raw
position rule differs.

* ``constant_short_vol`` — permanent short volatility. Pure carry: the
  variance risk premium collected with zero intelligence.
* ``contango_rule`` — short volatility only when the VIX term structure
  is in contango (VIX/VIX3M < threshold), flat otherwise. A documented
  zero-learned-parameter heuristic, and historically hard to beat.
* ``carry_with_kill_switch`` — permanent short vol, but flat whenever a
  model score says volatility is underpriced. Tests the ML forecast as
  a *risk filter* on the carry rather than as a standalone signal.
"""

from __future__ import annotations

import pandas as pd


def constant_short_vol(index: pd.Index) -> pd.Series:
    """Permanent -1 position (short volatility, pure carry)."""
    return pd.Series(-1.0, index=index, name="position")


def contango_rule(
    term_structure: pd.Series,
    threshold: float = 1.0,
) -> pd.Series:
    """Short vol when VIX/VIX3M < threshold (contango), flat otherwise."""
    position = pd.Series(0.0, index=term_structure.index, name="position")
    position[term_structure < threshold] = -1.0
    return position


def carry_with_kill_switch(
    score: pd.Series,
    score_threshold: float = 0.0,
) -> pd.Series:
    """Short vol carry, cut to flat when the model warns.

    ``score`` is log(predicted RV / implied vol): a value above
    ``score_threshold`` means the model expects more volatility than the
    market prices — the carry is switched off on those days.
    """
    position = pd.Series(-1.0, index=score.index, name="position")
    position[score > score_threshold] = 0.0
    return position


def carry_with_tail_switch(
    tail_probability: pd.Series,
    proba_cut: float = 0.25,
) -> pd.Series:
    """Short vol carry, cut to flat when P(tail event) exceeds the cut.

    ``tail_probability`` is a classifier's P(future RV > k * IV): the
    days most likely to hurt a short-vol position are skipped.
    """
    if not 0.0 <= proba_cut <= 1.0:
        raise ValueError(f"proba_cut must be in [0, 1], got {proba_cut}")
    position = pd.Series(-1.0, index=tail_probability.index, name="position")
    position[tail_probability > proba_cut] = 0.0
    return position


def combined_carry(
    score: pd.Series,
    term_structure: pd.Series,
    score_threshold: float = 0.0,
    contango_threshold: float = 1.0,
) -> pd.Series:
    """Carry cut by EITHER safety switch: model warning or backwardation.

    Short vol only when the model does not expect underpriced volatility
    (``score <= score_threshold``) AND the VIX term structure is in
    contango (``VIX/VIX3M < contango_threshold``). The two switches are
    complementary: the model selects bad *days*, the term structure cuts
    exposure *during* stress regimes the model cannot see coming.
    """
    aligned_score, aligned_ts = score.align(term_structure, join="inner")
    position = pd.Series(-1.0, index=aligned_score.index, name="position")
    position[aligned_score > score_threshold] = 0.0
    position[aligned_ts >= contango_threshold] = 0.0
    return position

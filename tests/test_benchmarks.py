"""Tests for the model-free benchmark strategies."""

from __future__ import annotations

import pandas as pd

from src.signals.benchmarks import (
    carry_with_kill_switch,
    constant_short_vol,
    contango_rule,
)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_constant_short_vol_is_always_minus_one() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    position = constant_short_vol(index)
    assert (position == -1.0).all()
    assert position.index.equals(index)


def test_contango_rule_short_only_in_contango() -> None:
    term_structure = _series([0.90, 0.99, 1.00, 1.10])
    position = contango_rule(term_structure, threshold=1.0)
    assert position.tolist() == [-1.0, -1.0, 0.0, 0.0]


def test_kill_switch_cuts_carry_on_model_warning() -> None:
    # score = log(predicted RV / IV): positive means vol underpriced.
    score = _series([-0.20, 0.05, -0.01, 0.30])
    position = carry_with_kill_switch(score, score_threshold=0.0)
    assert position.tolist() == [-1.0, 0.0, -1.0, 0.0]

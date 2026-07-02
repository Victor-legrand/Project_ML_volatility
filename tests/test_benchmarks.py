"""Tests for the model-free benchmark strategies."""

from __future__ import annotations

import pandas as pd

from src.signals.benchmarks import (
    carry_with_kill_switch,
    combined_carry,
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


def test_combined_carry_requires_both_switches_off() -> None:
    # Short only when score <= 0 (no model warning) AND contango (< 1).
    score = _series([-0.20, 0.10, -0.20, 0.10])
    term_structure = _series([0.95, 0.95, 1.05, 1.05])
    position = combined_carry(score, term_structure)
    assert position.tolist() == [-1.0, 0.0, 0.0, 0.0]

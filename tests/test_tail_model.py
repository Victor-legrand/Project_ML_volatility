"""Tests for the tail-event classifier building blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.tail_model import (
    ProbabilityClassifier,
    build_tail_logistic,
    decile_lift,
    tail_labels,
)
from src.models.walkforward import walk_forward_predictions


def _index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def test_tail_labels_threshold() -> None:
    future_rv = pd.Series([0.10, 0.31, 0.20], index=_index(3))
    implied = pd.Series([0.20, 0.20, 0.20], index=_index(3))
    labels = tail_labels(future_rv, implied, ratio_threshold=1.5)
    # Event iff future RV > 0.30.
    assert labels.tolist() == [0, 1, 0]


def test_probability_classifier_outputs_probabilities() -> None:
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"x": rng.normal(size=n)}, index=_index(n))
    y = (X["x"] > 1.0).astype(int)  # ~16% positives

    model = build_tail_logistic({"C": 1.0})
    model.fit(X, y)
    proba = model.predict(X)

    assert ((proba >= 0) & (proba <= 1)).all()
    # Higher x must rank as more likely tail.
    assert proba[X["x"].argmax()] > proba[X["x"].argmin()]


def test_probability_classifier_handles_single_class_window() -> None:
    X = pd.DataFrame({"x": np.arange(50.0)}, index=_index(50))
    y = pd.Series(0, index=X.index)  # no tail event in the window

    model = build_tail_logistic({"C": 1.0})
    model.fit(X, y)
    assert (model.predict(X) == 0).all()


def test_classifier_plugs_into_walkforward() -> None:
    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame({"x": rng.normal(size=n)}, index=_index(n))
    y = (X["x"] > 0.8).astype(int)

    probabilities, _ = walk_forward_predictions(
        X, y, {"clf": lambda: build_tail_logistic({"C": 1.0})},
        train_window=100, refit_every=20, purge=5,
    )
    values = probabilities["clf"]
    assert ((values >= 0) & (values <= 1)).all()
    assert len(values) == n - 105


def test_decile_lift_concentrates_events() -> None:
    n = 1000
    labels = pd.Series([1 if i >= 900 else 0 for i in range(n)], index=_index(n))
    proba = pd.Series(np.linspace(0, 1, n), index=_index(n))  # perfect ranking
    lift = decile_lift(labels, proba)
    assert lift.loc[10, "tail_frequency"] == 1.0
    assert lift.loc[1, "tail_frequency"] == 0.0
    assert lift.loc[10, "lift"] == 10.0

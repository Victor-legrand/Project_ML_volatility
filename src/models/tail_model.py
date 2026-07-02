"""Tail-event classification: predicting volatility shocks.

The strategy benchmarks showed that a *level* forecast of volatility
adds little protection during shocks that start from calm regimes
(Feb 2018, Aug 2024): by the time the level forecast reacts, the term
structure has already inverted. This module reframes the problem as
binary classification of the event that actually hurts a short-vol
carry position:

    tail_t = 1  iff  RV_{t+1..t+h} > ratio_threshold * IV_t,

i.e. realized volatility blowing through the level the option market
priced. Classifiers output a probability, evaluated with ranking and
calibration metrics (AUC, Brier, decile lift) and usable as a
de-risking switch on the carry.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def tail_labels(
    future_rv: pd.Series,
    implied_vol: pd.Series,
    ratio_threshold: float = 1.5,
) -> pd.Series:
    """Binary labels: future RV exceeds ratio_threshold times implied vol."""
    if ratio_threshold <= 0:
        raise ValueError(f"ratio_threshold must be > 0, got {ratio_threshold}")
    aligned_rv, aligned_iv = future_rv.align(implied_vol, join="inner")
    return (aligned_rv > ratio_threshold * aligned_iv).astype(int).rename("tail")


class ProbabilityClassifier:
    """Adapter: sklearn classifier whose ``predict`` returns P(class=1).

    Lets a classifier plug into ``walk_forward_predictions`` unchanged,
    since that protocol only calls ``fit`` and ``predict``. A training
    window containing a single class (a calm 5-year stretch can have no
    tail event at all) yields that class's probability everywhere,
    since sklearn refuses to fit on one class.
    """

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator
        self._single_class: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProbabilityClassifier":
        unique_classes = pd.unique(y)
        if len(unique_classes) == 1:
            self._single_class = float(unique_classes[0])
            return self
        self._single_class = None
        self._estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._single_class is not None:
            return np.full(len(X), self._single_class)
        probabilities = self._estimator.predict_proba(X)
        classes = list(self._estimator.classes_)
        return probabilities[:, classes.index(1)]


def build_tail_logistic(params: dict[str, Any]) -> ProbabilityClassifier:
    """Standardized logistic regression with balanced class weights."""
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5_000, **params),
    )
    return ProbabilityClassifier(estimator)


def build_tail_gradient_boosting(params: dict[str, Any]) -> ProbabilityClassifier:
    """Gradient boosting classifier (probabilities via log-loss)."""
    return ProbabilityClassifier(GradientBoostingClassifier(**params))


def build_tail_random_forest(params: dict[str, Any]) -> ProbabilityClassifier:
    """Random forest classifier with balanced class weights."""
    return ProbabilityClassifier(
        RandomForestClassifier(class_weight="balanced_subsample", **params)
    )


def evaluate_tail_probabilities(
    labels: pd.Series,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """AUC, Brier score and base-rate comparison for each model.

    ``brier_base`` is the Brier score of always predicting the base
    rate: a useful model must do better.
    """
    base_rate = float(labels.mean())
    brier_base = float(((labels - base_rate) ** 2).mean())
    rows = {}
    for name in probabilities.columns:
        proba = probabilities[name]
        rows[name] = {
            "auc": float(roc_auc_score(labels, proba)),
            "brier": float(brier_score_loss(labels, proba)),
            "brier_base": brier_base,
            "base_rate": base_rate,
        }
    return pd.DataFrame(rows).T.sort_values("auc", ascending=False)


def decile_lift(labels: pd.Series, probabilities: pd.Series) -> pd.DataFrame:
    """Observed tail frequency per predicted-probability decile.

    A useful classifier concentrates the events in the top deciles;
    ``lift`` is the ratio of each decile's frequency to the base rate.
    """
    deciles = pd.qcut(probabilities.rank(method="first"), 10, labels=False) + 1
    table = pd.DataFrame({"decile": deciles, "label": labels})
    grouped = table.groupby("decile")["label"].agg(["mean", "count"])
    grouped.columns = ["tail_frequency", "n_days"]
    grouped["lift"] = grouped["tail_frequency"] / labels.mean()
    return grouped

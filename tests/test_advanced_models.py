"""Tests for the v4 modelling additions (tuning, stacking, expanding WF)."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd

from src.models.ml_model import (
    build_stacking_meta_model,
    build_tuned_hist_gradient_boosting,
)
from src.models.walkforward import walk_forward_predictions


def _index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-01", periods=n)


def _tuning_config(n_iter: int = 4) -> dict:
    return {
        "tuning_iterations": n_iter,
        "cv_splits": 3,
        "random_state": 0,
        "param_grid": {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [7, 15],
            "max_iter": [30, 60],
        },
    }


def test_tuned_model_fits_predicts_and_records_params() -> None:
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"x": rng.normal(size=n)}, index=_index(n))
    y = X["x"] * 2.0 + rng.normal(0, 0.1, n)

    model = build_tuned_hist_gradient_boosting(_tuning_config())
    model.fit(X, y)

    assert model.best_params_ is not None
    assert set(model.best_params_) == {"learning_rate", "max_leaf_nodes", "max_iter"}
    predictions = model.predict(X)
    assert np.corrcoef(predictions, y)[0, 1] > 0.9


def test_tuned_model_is_picklable_after_fit() -> None:
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)}, index=_index(n))
    y = X["x"] + rng.normal(0, 0.1, n)
    model = build_tuned_hist_gradient_boosting(_tuning_config(n_iter=2)).fit(X, y)

    buffer = io.BytesIO()
    joblib.dump(model, buffer)   # must not raise (daily job reloads models)
    buffer.seek(0)
    reloaded = joblib.load(buffer)
    np.testing.assert_allclose(reloaded.predict(X), model.predict(X))


def test_walkforward_min_train_enables_expanding_window() -> None:
    n, min_train, purge = 200, 50, 5
    X = pd.DataFrame({"x": np.arange(n, dtype=float)}, index=_index(n))
    y = pd.Series(np.arange(n, dtype=float), index=_index(n))

    predictions, _ = walk_forward_predictions(
        X, y, {"m": lambda: build_stacking_meta_model(1.0)},
        train_window=n, refit_every=20, purge=purge, min_train=min_train,
    )
    # First prediction right after min_train + purge, not train_window.
    assert len(predictions) == n - (min_train + purge)


def test_stacking_meta_model_weights_are_non_negative() -> None:
    rng = np.random.default_rng(2)
    n = 400
    truth = pd.Series(rng.normal(size=n), index=_index(n))
    base = pd.DataFrame(
        {
            "good": truth + rng.normal(0, 0.1, n),
            "bad": -truth + rng.normal(0, 0.1, n),   # anti-correlated forecast
        },
        index=_index(n),
    )
    meta = build_stacking_meta_model(1.0).fit(base, truth)
    assert (meta.coef_ >= 0).all()
    assert meta.coef_[0] > 0.5   # weight goes to the good forecaster

"""Walk-forward (rolling re-training) prediction protocol.

Instead of a single train/test split, every model is re-fitted on a
rolling window and used to predict only the next block of dates. A purge
gap of ``purge`` days is left between the last training date and the
first prediction date: the target at date ``t`` uses returns up to
``t + horizon``, so training on dates within ``horizon`` days of the
prediction block would leak future information (target overlap).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

ModelFactory = Callable[[], Any]


def walk_forward_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    factories: dict[str, ModelFactory],
    train_window: int = 1260,
    refit_every: int = 63,
    purge: int = 5,
    min_train: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate out-of-sample predictions for several models.

    Parameters
    ----------
    X, y:
        Full feature matrix and target, chronologically ordered.
    factories:
        Mapping name -> zero-argument callable returning a fresh
        (unfitted) model with ``fit``/``predict``.
    train_window:
        Rolling training window length in rows (~trading days).
    refit_every:
        Number of rows predicted by each fitted model before refitting.
    purge:
        Gap (rows) between the end of the training set and the first
        predicted row. Must be >= the target horizon.
    min_train:
        Rows required before the first prediction. Defaults to
        ``train_window``; set it lower (with a large ``train_window``)
        for an expanding-window protocol — used by the stacking meta
        model, which starts from little out-of-sample history.

    Returns
    -------
    (predictions, last_models):
        ``predictions`` has one column per model, indexed on the
        predicted dates only. ``last_models`` holds the final fitted
        model of each family (for inspection / feature importances).
    """
    if purge < 0:
        raise ValueError(f"purge must be >= 0, got {purge}")
    n_rows = len(X)
    first_prediction = (min_train if min_train is not None else train_window) + purge
    if first_prediction >= n_rows:
        raise ValueError(
            f"Not enough data: {n_rows} rows for min_train="
            f"{min_train if min_train is not None else train_window} and purge={purge}"
        )

    predictions: dict[str, pd.Series] = {
        name: pd.Series(np.nan, index=X.index) for name in factories
    }
    last_models: dict[str, Any] = {}

    for block_start in range(first_prediction, n_rows, refit_every):
        block_end = min(block_start + refit_every, n_rows)
        train_end = block_start - purge
        train_start = max(0, train_end - train_window)
        X_train, y_train = X.iloc[train_start:train_end], y.iloc[train_start:train_end]
        X_block = X.iloc[block_start:block_end]

        for name, factory in factories.items():
            model = factory()
            model.fit(X_train, y_train)
            block_pred = np.asarray(model.predict(X_block)).ravel()
            predictions[name].iloc[block_start:block_end] = block_pred
            last_models[name] = model

    result = pd.DataFrame(predictions).dropna(how="any")
    return result, last_models

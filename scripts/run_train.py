"""Step 3: walk-forward training and comparison of all forecasting models.

Models: HAR-RV, HAR-X, Ridge, Lasso, Random Forest, Gradient Boosting,
quantile GB (q10/q90), an ensemble, and two naive benchmarks
("prediction = current RV" and "prediction = implied vol"). Every model
is re-fitted on a rolling purged window (walk-forward), predictions are
converted back to RV space and compared with RMSE / MAE / R², per-year
RMSE and Diebold-Mariano tests.

Usage: python scripts/run_train.py (requires run_features.py first)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.features.volatility_features import invert_target, split_features_target
from src.models.baseline import HARModel, HARXModel
from src.models.evaluation import dm_table, evaluation_table, rmse_by_period
from src.models.ml_model import (
    build_extra_trees,
    build_gradient_boosting,
    build_lasso,
    build_quantile_gradient_boosting,
    build_random_forest,
    build_ridge,
    build_stacking_meta_model,
    build_tuned_hist_gradient_boosting,
    feature_importances,
)
from src.models.walkforward import walk_forward_predictions
from src.utils.io import load_config, load_dataframe, resolve_path, save_dataframe

QUANTILE_PREFIX = "gb_q"


def build_factories(models_cfg: dict, vol_windows: list[int]) -> dict:
    """Zero-argument factories for every walk-forward model."""
    factories = {
        "har": lambda: HARModel(vol_windows),
        "harx": lambda: HARXModel(vol_windows),
        "ridge": lambda: build_ridge(models_cfg["ridge"]),
        "lasso": lambda: build_lasso(models_cfg["lasso"]),
        "random_forest": lambda: build_random_forest(models_cfg["random_forest"]),
        "gradient_boosting": lambda: build_gradient_boosting(
            models_cfg["gradient_boosting"]
        ),
        "extra_trees": lambda: build_extra_trees(models_cfg["extra_trees"]),
        "hist_gb": lambda: build_tuned_hist_gradient_boosting(
            models_cfg["hist_gradient_boosting"]
        ),
    }
    gb_params = models_cfg["gradient_boosting"]
    for quantile in models_cfg["quantiles"]:
        name = f"{QUANTILE_PREFIX}{int(quantile * 100)}"
        factories[name] = (
            lambda q=quantile: build_quantile_gradient_boosting(gb_params, q)
        )
    return factories


def main() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    models_cfg = config["models"]
    feat_cfg = config["features"]
    wf_cfg = models_cfg["walk_forward"]
    target_type = feat_cfg["target_type"]
    horizon = feat_cfg["target_horizon"]

    dataset = load_dataframe(processed_dir / "features.csv")
    X, y, future_rv = split_features_target(dataset)
    implied_vol = X["implied_vol"]

    factories = build_factories(models_cfg, feat_cfg["vol_windows"])
    print(f"Walk-forward: window={wf_cfg['train_window']}d, "
          f"refit every {wf_cfg['refit_every']}d, purge={wf_cfg['purge']}d, "
          f"{len(factories)} models, target_type={target_type}")
    raw_predictions, last_models = walk_forward_predictions(
        X, y, factories,
        train_window=wf_cfg["train_window"],
        refit_every=wf_cfg["refit_every"],
        purge=wf_cfg["purge"],
    )
    print(f"Out-of-sample: {raw_predictions.index[0].date()} -> "
          f"{raw_predictions.index[-1].date()} ({len(raw_predictions)} obs)")

    # Stacking: a meta model fitted on the *accumulated out-of-sample*
    # predictions of the point models (expanding window, same purge).
    # Leak-free by construction: every meta training row is a genuine
    # past out-of-sample forecast.
    stack_cfg = models_cfg["stacking"]
    base_columns = [c for c in raw_predictions.columns
                    if not c.startswith(QUANTILE_PREFIX)]
    stacked_raw, stack_models = walk_forward_predictions(
        raw_predictions[base_columns],
        y.reindex(raw_predictions.index),
        {"stacked": lambda: build_stacking_meta_model(stack_cfg["meta_alpha"])},
        train_window=len(raw_predictions),      # expanding window
        refit_every=wf_cfg["refit_every"],
        purge=wf_cfg["purge"],
        min_train=stack_cfg["min_train"],
    )
    raw_predictions = raw_predictions.join(stacked_raw)
    last_models["stacked"] = stack_models["stacked"]

    # Back to RV space so all models and benchmarks are comparable.
    predictions_rv = raw_predictions.apply(
        lambda col: invert_target(col, implied_vol, target_type)
    )
    predictions_rv["ensemble"] = predictions_rv[models_cfg["ensemble_members"]].mean(axis=1)
    predictions_rv["naive_rw"] = X[f"rv_{horizon}"].reindex(predictions_rv.index)
    predictions_rv["naive_implied"] = implied_vol.reindex(predictions_rv.index)
    y_true_rv = future_rv.reindex(predictions_rv.index)

    point_forecasts = predictions_rv.drop(
        columns=[c for c in predictions_rv.columns if c.startswith(QUANTILE_PREFIX)]
    )
    # The stacked model starts min_train days later: compare everyone on
    # the common sample where it exists.
    common = point_forecasts.dropna().index
    print(f"\nOut-of-sample comparison in RV space, common sample "
          f"({common[0].date()} -> {common[-1].date()}, {len(common)} obs):")
    print(evaluation_table(y_true_rv.loc[common], point_forecasts.loc[common])
          .round(4).to_string())
    weights = pd.Series(
        last_models["stacked"].coef_, index=base_columns
    ).round(3)
    print(f"\nStacking weights (last refit): {weights[weights > 0].to_dict()}")

    print("\nRMSE by year (stability across regimes, common sample):")
    print(rmse_by_period(y_true_rv.loc[common], point_forecasts.loc[common])
          .round(4).to_string())

    print("\nDiebold-Mariano vs HAR (dm_stat < 0 => beats HAR):")
    print(dm_table(y_true_rv.loc[common], point_forecasts.loc[common],
                   benchmark="har", horizon=horizon).round(4).to_string())

    models_dir = resolve_path(processed_dir / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in last_models.items():
        joblib.dump(model, models_dir / f"{name}.joblib")

    rf = last_models["random_forest"]
    print("\nRandom Forest (last refit) — top feature importances:")
    print(feature_importances(rf, list(X.columns)).head(10).round(3).to_string())

    predictions_rv["y_true_rv"] = y_true_rv
    save_dataframe(predictions_rv, processed_dir / "predictions_rv.csv")
    print(f"\nModels saved in {models_dir}")
    print(f"Predictions (RV space) saved: {processed_dir / 'predictions_rv.csv'}")


if __name__ == "__main__":
    main()

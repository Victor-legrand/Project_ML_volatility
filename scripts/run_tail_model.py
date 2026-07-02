"""Step 6: walk-forward tail classifier — P(future RV > k * implied vol).

Trains logistic regression, gradient boosting and random forest
classifiers on the same purged walk-forward protocol as the level
models, evaluates them with AUC / Brier / decile lift, and saves the
probabilities for use as a de-risking switch in the strategy benchmarks.

Usage: python scripts/run_tail_model.py (requires run_features.py first)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.features.volatility_features import split_features_target
from src.models.tail_model import (
    build_tail_gradient_boosting,
    build_tail_logistic,
    build_tail_random_forest,
    decile_lift,
    evaluate_tail_probabilities,
    tail_labels,
)
from src.models.walkforward import walk_forward_predictions
from src.utils.io import load_config, load_dataframe, save_dataframe


def main() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    tail_cfg = config["tail_model"]
    wf_cfg = config["models"]["walk_forward"]

    dataset = load_dataframe(processed_dir / "features.csv")
    X, _, future_rv = split_features_target(dataset)
    labels = tail_labels(future_rv, X["implied_vol"], tail_cfg["ratio_threshold"])
    print(f"Tail event: future RV > {tail_cfg['ratio_threshold']} x implied vol")
    print(f"Base rate: {labels.mean():.2%} of days ({labels.sum()} events "
          f"/ {len(labels)} days)")

    factories = {
        "tail_logistic": lambda: build_tail_logistic(tail_cfg["logistic"]),
        "tail_gb": lambda: build_tail_gradient_boosting(tail_cfg["gradient_boosting"]),
        "tail_rf": lambda: build_tail_random_forest(tail_cfg["random_forest"]),
    }
    probabilities, _ = walk_forward_predictions(
        X, labels, factories,
        train_window=wf_cfg["train_window"],
        refit_every=wf_cfg["refit_every"],
        purge=wf_cfg["purge"],
    )
    oos_labels = labels.reindex(probabilities.index)
    print(f"Out-of-sample: {probabilities.index[0].date()} -> "
          f"{probabilities.index[-1].date()} ({len(probabilities)} obs, "
          f"base rate {oos_labels.mean():.2%})")

    print("\nClassifier comparison (AUC: ranking, Brier: calibration):")
    print(evaluate_tail_probabilities(oos_labels, probabilities).round(4).to_string())

    best = tail_cfg["model_name"]
    print(f"\nDecile lift — {best} (top decile should concentrate the shocks):")
    print(decile_lift(oos_labels, probabilities[best]).round(3).to_string())

    cut = tail_cfg["proba_cut"]
    flagged = probabilities[best] > cut
    if flagged.any():
        capture = oos_labels[flagged].sum() / max(oos_labels.sum(), 1)
        print(f"\nAt proba_cut={cut}: {flagged.mean():.1%} of days flagged, "
              f"capturing {capture:.1%} of tail events "
              f"(precision {oos_labels[flagged].mean():.1%}).")
    else:
        print(f"\nAt proba_cut={cut}: no day flagged — consider lowering the cut.")

    output = probabilities.copy()
    output["tail_label"] = oos_labels
    save_dataframe(output, processed_dir / "tail_probabilities.csv")
    print(f"\nTail probabilities saved: {processed_dir / 'tail_probabilities.csv'}")


if __name__ == "__main__":
    main()

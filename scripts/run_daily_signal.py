"""Daily signal job: fresh data -> today's recommendation -> notification.

Downloads the latest daily data, rebuilds today's features, scores them
with the models saved by ``run_train.py``, applies the three-layer rule
(carry + contango filter + ML kill switch), appends the result to
``data/processed/signal_log.csv`` (the paper-trading audit trail) and
sends the report on the configured channels.

This job recommends; it never executes orders.

Usage: python scripts/run_daily_signal.py   (requires run_train.py once,
and again quarterly to refresh the saved models)

Schedule it after the US close (22:15 Paris), e.g. with cron:
    15 22 * * 1-5  cd /path/to/vol_ml_fund && .venv/bin/python scripts/run_daily_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.data.download import download_daily_data
from src.data.preprocess import clean_prices, compute_log_returns
from src.features.volatility_features import (
    NON_FEATURE_COLUMNS,
    build_feature_matrix,
    invert_target,
)
from src.signals.recommendation import format_message, recommend_position
from src.utils.io import load_config, resolve_path, save_dataframe
from src.utils.notify import dispatch


def latest_features(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download fresh data and build the feature matrix up to today."""
    data_cfg = config["data"]
    feat_cfg = config["features"]
    required = data_cfg["core_tickers"] + data_cfg["aux_tickers"]
    tickers = required + data_cfg["proxy_tickers"]
    close, ohlc = download_daily_data(tickers, data_cfg["start_date"])
    prices = clean_prices(close, required_columns=required)
    returns = compute_log_returns(prices)
    dataset = build_feature_matrix(
        returns=returns[data_cfg["core_tickers"]],
        prices=prices,
        target_ohlc=ohlc[feat_cfg["target_ticker"]],
        target_ticker=feat_cfg["target_ticker"],
        implied_vol_index=prices["^VIX"],
        implied_vol_3m_index=prices["^VIX3M"],
        vol_windows=feat_cfg["vol_windows"],
        gk_windows=feat_cfg["gk_windows"],
        target_horizon=feat_cfg["target_horizon"],
        target_type=feat_cfg["target_type"],
        shock_sigma_window=feat_cfg["shock_sigma_window"],
        shock_threshold=feat_cfg["shock_threshold"],
        trading_days_per_year=feat_cfg["trading_days_per_year"],
        require_target=False,   # today's future window is not observed yet
        aux_indices=prices[data_cfg["aux_tickers"]],
    )
    return dataset, prices, returns


def predict_rv_with_model(
    model_name: str,
    X_today: pd.DataFrame,
    implied_vol: pd.Series,
    config: dict,
) -> float:
    """Today's predicted RV (RV space) from one saved model."""
    models_dir = resolve_path(Path(config["data"]["processed_dir"]) / "models")
    model_path = models_dir / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing saved model {model_path} — run scripts/run_train.py first."
        )
    model = joblib.load(model_path)
    raw = pd.Series(np.asarray(model.predict(X_today)).ravel(), index=X_today.index)
    inverted = invert_target(raw, implied_vol, config["features"]["target_type"])
    return float(inverted.iloc[-1])


def ensemble_prediction(
    X_today: pd.DataFrame,
    implied_vol: pd.Series,
    config: dict,
) -> float:
    """Predicted future RV from the saved ensemble members (RV space)."""
    members = config["models"]["ensemble_members"]
    values = [
        predict_rv_with_model(name, X_today, implied_vol, config)
        for name in members
    ]
    return float(np.mean(values))


def append_signal_log(log_path: Path, row: dict) -> None:
    """Upsert today's row in the signal log (idempotent re-runs)."""
    full_path = resolve_path(log_path)
    if full_path.exists():
        log = pd.read_csv(full_path, index_col=0, parse_dates=True)
    else:
        log = pd.DataFrame()
    date_index = pd.Timestamp(row.pop("date"))
    log.loc[date_index, list(row)] = list(row.values())
    log.index.name = "date"
    save_dataframe(log.sort_index(), log_path)


def main() -> None:
    config = load_config()
    daily_cfg = config["daily_signal"]
    signal_cfg = config["signal"]
    bt_cfg = config["backtest"]

    print("Downloading fresh data and building today's features...")
    dataset, prices, _ = latest_features(config)
    X = dataset.drop(columns=NON_FEATURE_COLUMNS)
    today = X.index[-1]

    stale_warning = None
    staleness = int(np.busday_count(today.date(), pd.Timestamp.now().date()))
    if staleness > daily_cfg["stale_after_days"]:
        stale_warning = (
            f"Dernière donnée : {today.date()} ({staleness} jours ouvrés de "
            "retard) — signal possiblement obsolète."
        )

    predicted_rv = ensemble_prediction(X.tail(1), X["implied_vol"], config)
    implied = float(X["implied_vol"].iloc[-1])
    score = float(np.log(predicted_rv / implied))
    term_structure = float(X["vix_term_structure"].iloc[-1])

    # Upper-quantile scenario, always computed for the risk display.
    q90_rv = predict_rv_with_model("gb_q90", X.tail(1), X["implied_vol"], config)
    q90_score = float(np.log(q90_rv / implied))

    # The kill switch can run on the upper-quantile scenario rather than
    # the mean forecast (see config daily_signal.kill_switch_model).
    switch_model = daily_cfg["kill_switch_model"]
    if switch_model == "ensemble":
        switch_rv, switch_score = predicted_rv, score
    elif switch_model == "gb_q90":
        switch_rv, switch_score = q90_rv, q90_score
    else:
        switch_rv = predict_rv_with_model(
            switch_model, X.tail(1), X["implied_vol"], config
        )
        switch_score = float(np.log(switch_rv / implied))

    short_ticker = bt_cfg["short_vol_ticker"]
    short_leg_returns = prices[short_ticker].pct_change()
    short_leg_vol = float(
        short_leg_returns.rolling(signal_cfg["proxy_vol_window"]).std().iloc[-1]
        * np.sqrt(config["features"]["trading_days_per_year"])
    )

    recommendation = recommend_position(
        score=switch_score,
        term_structure=term_structure,
        short_leg_vol=short_leg_vol,
        score_threshold=daily_cfg["score_threshold"],
        contango_threshold=daily_cfg["contango_threshold"],
        vol_target=signal_cfg["vol_target"],
        max_leverage=signal_cfg["max_leverage"],
    )

    metrics = {
        "vix": implied * 100.0,
        "vix3m": implied * 100.0 / term_structure,
        "term_structure": term_structure,
        "predicted_rv": predicted_rv,
        "implied_vol": implied,
        "score": score,
        "q90_rv": q90_rv,
        "q90_score": q90_score,
        "switch_model": switch_model,
        "switch_rv": switch_rv,
        "switch_score": switch_score,
        "short_leg_ticker": short_ticker,
    }
    message = format_message(today.date(), metrics, recommendation, stale_warning)

    log_path = Path(config["data"]["processed_dir"]) / "signal_log.csv"
    append_signal_log(log_path, {
        "date": today,
        "vix": metrics["vix"],
        "term_structure": term_structure,
        "predicted_rv": predicted_rv,
        "score": score,
        "q90_score": q90_score,
        "switch_model": switch_model,
        "switch_score": switch_score,
        "short_leg_vol": short_leg_vol,
        "stance": recommendation.stance,
        "scaled_position": recommendation.scaled_position,
    })

    statuses = dispatch(message, daily_cfg["channels"])
    print(f"\nSignal logged in {log_path}")
    for channel, status in statuses.items():
        marker = "OK" if status == "ok" else f"FAILED ({status})"
        print(f"Notification {channel}: {marker}")


if __name__ == "__main__":
    main()

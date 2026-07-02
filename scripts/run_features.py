"""Step 2: build the volatility feature matrix and the prediction target.

Usage: python scripts/run_features.py (requires run_download.py first)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.volatility_features import build_feature_matrix
from src.utils.io import load_config, load_dataframe, save_dataframe


def main() -> None:
    config = load_config()
    data_cfg = config["data"]
    feat_cfg = config["features"]
    processed_dir = Path(data_cfg["processed_dir"])
    raw_dir = Path(data_cfg["raw_dir"])

    prices = load_dataframe(processed_dir / "prices.csv")
    returns = load_dataframe(processed_dir / "log_returns.csv")
    target_ticker = feat_cfg["target_ticker"]
    target_ohlc = load_dataframe(raw_dir / f"ohlc_{target_ticker.lower()}.csv")

    dataset = build_feature_matrix(
        returns=returns[data_cfg["core_tickers"]],
        prices=prices,
        target_ohlc=target_ohlc,
        target_ticker=target_ticker,
        implied_vol_index=prices["^VIX"],
        implied_vol_3m_index=prices["^VIX3M"],
        vol_windows=feat_cfg["vol_windows"],
        gk_windows=feat_cfg["gk_windows"],
        target_horizon=feat_cfg["target_horizon"],
        target_type=feat_cfg["target_type"],
        shock_sigma_window=feat_cfg["shock_sigma_window"],
        shock_threshold=feat_cfg["shock_threshold"],
        trading_days_per_year=feat_cfg["trading_days_per_year"],
    )

    dataset_path = save_dataframe(dataset, processed_dir / "features.csv")
    n_features = dataset.shape[1] - 2  # minus target and future_rv
    print(f"Feature matrix saved: {dataset_path}")
    print(f"  {len(dataset)} rows, {n_features} features "
          f"(target_type={feat_cfg['target_type']})")
    print(f"  columns: {list(dataset.columns)}")


if __name__ == "__main__":
    main()

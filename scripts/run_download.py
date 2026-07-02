"""Step 1: download raw daily OHLC data, save cleaned prices + log returns.

Usage: python scripts/run_download.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import download_daily_data
from src.data.preprocess import clean_prices, compute_log_returns
from src.utils.io import load_config, save_dataframe


def safe_name(ticker: str) -> str:
    """Filesystem-friendly ticker name (^VIX -> vix)."""
    return ticker.replace("^", "").lower()


def main() -> None:
    config = load_config()
    data_cfg = config["data"]
    core = data_cfg["core_tickers"]
    proxies = data_cfg["proxy_tickers"]
    tickers = core + proxies

    print(f"Downloading {tickers} from {data_cfg['start_date']}...")
    close, ohlc = download_daily_data(
        tickers=tickers,
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
    )
    raw_dir = Path(data_cfg["raw_dir"])
    save_dataframe(close, raw_dir / "prices_raw.csv")
    for ticker, frame in ohlc.items():
        save_dataframe(frame, raw_dir / f"ohlc_{safe_name(ticker)}.csv")
    print(f"Raw data saved in {raw_dir} ({len(close)} rows)")

    # Core tickers are required on every row; tradable proxies (VIXY,
    # SVXY) start later and keep their leading NaNs.
    cleaned = clean_prices(close, required_columns=core)
    returns = compute_log_returns(cleaned)

    processed_dir = Path(data_cfg["processed_dir"])
    save_dataframe(cleaned, processed_dir / "prices.csv")
    save_dataframe(returns, processed_dir / "log_returns.csv")
    print(f"Clean prices saved: {processed_dir / 'prices.csv'} ({len(cleaned)} rows)")
    print(f"Log returns saved:  {processed_dir / 'log_returns.csv'} ({len(returns)} rows)")
    print(f"Sample: {cleaned.index[0].date()} -> {cleaned.index[-1].date()}")


if __name__ == "__main__":
    main()

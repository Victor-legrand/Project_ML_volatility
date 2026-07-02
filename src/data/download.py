"""Download daily market data (OHLC) with yfinance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

OHLC_FIELDS = ["Open", "High", "Low", "Close"]


def _download_frame(
    tickers: list[str],
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    """Raw yfinance download with (field, ticker) MultiIndex columns."""
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")
    if not isinstance(raw.columns, pd.MultiIndex):  # single ticker
        raw.columns = pd.MultiIndex.from_product([raw.columns, [tickers[0]]])
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    return raw.sort_index()


def download_daily_data(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Download daily adjusted OHLC data for a list of tickers.

    Returns
    -------
    (close, ohlc):
        ``close`` is a wide dataframe of close prices (one column per
        ticker). ``ohlc`` maps each ticker to an Open/High/Low/Close
        dataframe. Prices are split/dividend adjusted.
    """
    raw = _download_frame(tickers, start_date, end_date)

    # A ticker can fail transiently (network, yfinance cache lock) and come
    # back as a missing or all-NaN column: retry those individually.
    close = raw["Close"]
    failed = [
        t for t in tickers if t not in close.columns or close[t].dropna().empty
    ]
    for ticker in failed:
        retry = _download_frame([ticker], start_date, end_date)
        if retry["Close"][ticker].dropna().empty:
            raise RuntimeError(f"No data downloaded for ticker: {ticker}")
        for field in OHLC_FIELDS:
            raw.loc[:, (field, ticker)] = retry[field][ticker]

    close = raw["Close"][tickers]
    ohlc = {
        ticker: raw.xs(ticker, axis=1, level=1)[OHLC_FIELDS].dropna(how="all")
        for ticker in tickers
    }
    return close, ohlc

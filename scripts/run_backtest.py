"""Step 4: signal (hysteresis + sizing + vol targeting), backtest, report.

The score log(predicted RV / implied vol) is traded through two
long-only legs: VIXY when long volatility, SVXY (inverse ETP) when
short volatility. Transaction costs apply to every weight change.

Usage: python scripts/run_backtest.py (requires run_train.py first)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, position_to_leg_weights, run_backtest
from src.backtest.metrics import performance_summary
from src.signals.volatility_signal import (
    conviction_size,
    hysteresis_direction,
    signal_statistics,
    vol_target_leverage,
    volatility_score,
)
from src.utils.io import load_config, load_dataframe, resolve_path, save_dataframe


def plot_report(
    result: BacktestResult, score: pd.Series, output_path: Path
) -> None:
    """Save a three-panel report: equity curve, net position, score."""
    fig, (ax_equity, ax_pos, ax_score) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True, height_ratios=[3, 1, 1]
    )
    ax_equity.plot(result.equity_curve.index, result.equity_curve.values, lw=1.2)
    ax_equity.set_title("Volatility strategy — equity curve (net of costs)")
    ax_equity.set_ylabel("Equity ($)")
    ax_equity.grid(alpha=0.3)

    ax_pos.fill_between(result.positions.index, result.positions.values,
                        step="mid", alpha=0.6)
    ax_pos.set_ylabel("Net position")
    ax_pos.grid(alpha=0.3)

    aligned_score = score.reindex(result.positions.index)
    ax_score.plot(aligned_score.index, aligned_score.values, lw=0.8, color="darkred")
    ax_score.axhline(0, color="k", lw=0.5)
    ax_score.set_ylabel("log(pred/IV)")
    ax_score.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    signal_cfg = config["signal"]
    bt_cfg = config["backtest"]

    predictions = load_dataframe(processed_dir / "predictions_rv.csv")
    features = load_dataframe(processed_dir / "features.csv")
    prices = load_dataframe(processed_dir / "prices.csv")

    model_name = signal_cfg["model_name"]
    if model_name not in predictions.columns:
        raise KeyError(
            f"Model '{model_name}' not in predictions: {list(predictions.columns)}"
        )
    implied_vol = features["implied_vol"]

    # 1. Score and quantile scores (monotonic transform preserves quantiles).
    score = volatility_score(predictions[model_name], implied_vol)
    score_q10 = score_q90 = None
    if signal_cfg["use_quantile_conviction"]:
        score_q10 = volatility_score(predictions["gb_q10"], implied_vol)
        score_q90 = volatility_score(predictions["gb_q90"], implied_vol)

    # 2-3. Hysteresis direction, conviction-based sizing.
    direction = hysteresis_direction(
        score, signal_cfg["entry_threshold"], signal_cfg["exit_threshold"]
    )
    position = conviction_size(
        score, direction, signal_cfg["sizing_scale"], score_q10, score_q90
    )

    # 4. Vol targeting on the traded leg (simple returns for compounding).
    leg_prices = prices[[bt_cfg["long_vol_ticker"], bt_cfg["short_vol_ticker"]]]
    leg_returns = leg_prices.pct_change()
    leg_returns.columns = ["long", "short"]
    position = vol_target_leverage(
        position,
        leg_returns,
        vol_target=signal_cfg["vol_target"],
        vol_window=signal_cfg["proxy_vol_window"],
        max_leverage=signal_cfg["max_leverage"],
    )

    weights = position_to_leg_weights(position)
    result = run_backtest(
        weights=weights,
        leg_returns=leg_returns,
        transaction_cost_bps=bt_cfg["transaction_cost_bps"],
        initial_capital=bt_cfg["initial_capital"],
    )

    summary = performance_summary(result.net_returns, result.equity_curve)
    stats = signal_statistics(result.positions)

    print(f"Signal model: {model_name} | legs: {bt_cfg['long_vol_ticker']} (long vol) "
          f"/ {bt_cfg['short_vol_ticker']} (short vol) "
          f"| costs: {bt_cfg['transaction_cost_bps']} bps one-way")
    print(f"Backtest period: {result.equity_curve.index[0].date()} -> "
          f"{result.equity_curve.index[-1].date()}")
    print("\nSignal statistics:")
    for key, value in stats.items():
        print(f"  {key:>20}: {value:.3f}")
    print("\nPerformance (net of costs):")
    for key, value in summary.items():
        print(f"  {key:>22}: {value:.4f}")
    total_costs = float(result.costs.sum())
    print(f"  {'cumulative costs':>22}: {total_costs:.4f} (sum of daily cost drags)")

    report_dir = resolve_path(bt_cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    backtest_df = pd.concat(
        [score.rename("score"), result.positions, result.net_returns,
         result.equity_curve],
        axis=1,
    ).dropna(subset=["equity"])
    save_dataframe(backtest_df, Path(bt_cfg["report_dir"]) / "backtest_timeseries.csv")
    pd.Series(summary).to_csv(report_dir / "performance_summary.csv", header=False)
    plot_report(result, score, report_dir / "equity_curve.png")
    print(f"\nReport saved in {report_dir}")


if __name__ == "__main__":
    main()

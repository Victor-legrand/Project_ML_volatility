"""Step 5: does the ML model earn its place in the portfolio?

Backtests, with the *same* engine, costs, vol targeting and period:

* constant_short   — permanent short vol (pure carry, zero intelligence)
* contango_rule    — short vol only when VIX/VIX3M < 1, flat otherwise
* ml_kill_switch   — carry, but flat when the model predicts RV > IV
* ml_strategy      — full ML signal (hysteresis + conviction sizing)

If ml_strategy does not clearly beat the first two (Sharpe AND drawdown,
especially through the stress windows), the forecasting pipeline adds no
portfolio value and the research should pivot to tail-event timing.

Usage: python scripts/run_strategy_benchmarks.py (requires run_train.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import position_to_leg_weights, run_backtest
from src.backtest.metrics import performance_summary
from src.signals.benchmarks import (
    carry_with_kill_switch,
    combined_carry,
    constant_short_vol,
    contango_rule,
)
from src.signals.volatility_signal import (
    conviction_size,
    hysteresis_direction,
    signal_statistics,
    vol_target_leverage,
    volatility_score,
)
from src.utils.io import load_config, load_dataframe, resolve_path, save_dataframe

STRESS_WINDOWS = {
    "volmageddon_2018": ("2018-01-15", "2018-03-01"),
    "covid_2020": ("2020-02-14", "2020-04-15"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "aug_2024_unwind": ("2024-07-15", "2024-08-15"),
}


def build_ml_position(
    predictions: pd.DataFrame,
    implied_vol: pd.Series,
    signal_cfg: dict,
) -> tuple[pd.Series, pd.Series]:
    """Full ML position (before vol targeting) and its score."""
    score = volatility_score(predictions[signal_cfg["model_name"]], implied_vol)
    score_q10 = score_q90 = None
    if signal_cfg["use_quantile_conviction"]:
        score_q10 = volatility_score(predictions["gb_q10"], implied_vol)
        score_q90 = volatility_score(predictions["gb_q90"], implied_vol)
    direction = hysteresis_direction(
        score, signal_cfg["entry_threshold"], signal_cfg["exit_threshold"]
    )
    position = conviction_size(
        score, direction, signal_cfg["sizing_scale"], score_q10, score_q90
    )
    return position, score


def window_returns(net_returns: pd.Series) -> dict[str, float]:
    """Total net return of the strategy over each stress window."""
    output: dict[str, float] = {}
    for label, (start, end) in STRESS_WINDOWS.items():
        window = net_returns.loc[start:end]
        output[label] = float((1.0 + window).prod() - 1.0) if len(window) else float("nan")
    return output


def main() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    signal_cfg = config["signal"]
    bt_cfg = config["backtest"]

    predictions = load_dataframe(processed_dir / "predictions_rv.csv")
    features = load_dataframe(processed_dir / "features.csv")
    prices = load_dataframe(processed_dir / "prices.csv")

    implied_vol = features["implied_vol"]
    leg_prices = prices[[bt_cfg["long_vol_ticker"], bt_cfg["short_vol_ticker"]]]
    leg_returns = leg_prices.pct_change()
    leg_returns.columns = ["long", "short"]

    # All strategies live on the ML out-of-sample dates: same period,
    # same information set, fair comparison.
    ml_position, score = build_ml_position(predictions, implied_vol, signal_cfg)
    common_index = ml_position.index
    term_structure = features["vix_term_structure"].reindex(common_index)

    raw_positions: dict[str, pd.Series] = {
        "constant_short": constant_short_vol(common_index),
        "contango_rule": contango_rule(term_structure),
        "ml_kill_switch": carry_with_kill_switch(score),
        "combined_carry": combined_carry(score, term_structure),
        "ml_strategy": ml_position,
    }

    summaries: dict[str, dict[str, float]] = {}
    stress: dict[str, dict[str, float]] = {}
    equity_curves: dict[str, pd.Series] = {}
    net_returns_all: dict[str, pd.Series] = {}

    for name, raw_position in raw_positions.items():
        position = vol_target_leverage(
            raw_position,
            leg_returns,
            vol_target=signal_cfg["vol_target"],
            vol_window=signal_cfg["proxy_vol_window"],
            max_leverage=signal_cfg["max_leverage"],
        )
        result = run_backtest(
            weights=position_to_leg_weights(position),
            leg_returns=leg_returns,
            transaction_cost_bps=bt_cfg["transaction_cost_bps"],
            initial_capital=bt_cfg["initial_capital"],
        )
        summary = performance_summary(result.net_returns, result.equity_curve)
        summary["avg_daily_turnover"] = signal_statistics(result.positions)[
            "avg_daily_turnover"
        ]
        summaries[name] = summary
        stress[name] = window_returns(result.net_returns)
        equity_curves[name] = result.equity_curve
        net_returns_all[name] = result.net_returns

    comparison = pd.DataFrame(summaries).T
    stress_table = pd.DataFrame(stress).T

    period = equity_curves["ml_strategy"]
    print(f"Same engine, costs ({bt_cfg['transaction_cost_bps']} bps), vol target "
          f"({signal_cfg['vol_target']:.0%}) and period for all strategies:")
    print(f"  {period.index[0].date()} -> {period.index[-1].date()} "
          f"({len(period)} days)\n")
    print("Strategy comparison (net of costs):")
    print(comparison.round(4).to_string())
    print("\nTotal net return during stress windows (short vol pain periods):")
    print(stress_table.round(4).to_string())

    carry_corr = net_returns_all["ml_strategy"].corr(net_returns_all["constant_short"])
    print(f"\nDaily return correlation ml_strategy vs constant_short: {carry_corr:.3f}")
    print("(a correlation near 1 means the ML strategy is repackaged carry)")

    report_dir = resolve_path(bt_cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(report_dir / "strategy_benchmarks.csv")
    stress_table.to_csv(report_dir / "strategy_stress_windows.csv")

    normalized = pd.DataFrame(
        {name: curve / curve.iloc[0] for name, curve in equity_curves.items()}
    )
    save_dataframe(normalized, Path(bt_cfg["report_dir"]) / "benchmark_equity_curves.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in normalized.columns:
        ax.plot(normalized.index, normalized[name], lw=1.1, label=name)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (normalized, log scale)")
    ax.set_title("ML strategy vs model-free benchmarks (same engine and costs)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "benchmark_equity_curves.png", dpi=120)
    plt.close(fig)
    print(f"\nReport saved in {report_dir}")


if __name__ == "__main__":
    main()

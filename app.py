"""Interactive research dashboard (Streamlit).

Sensitivity-analysis tool on top of the pipeline outputs: the signal
parameters can be changed live and the backtest is recomputed through
the exact same engine as ``scripts/run_backtest.py`` — no duplicated
logic, so the dashboard cannot drift from the scripts.

Run from the project root, after the pipeline:
    streamlit run app.py

This dashboard reads ``data/processed`` only; it never launches
downloads or training, and it executes no orders.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.backtest.engine import BacktestResult, position_to_leg_weights, run_backtest
from src.backtest.metrics import performance_summary
from src.models.evaluation import dm_table, evaluation_table, rmse_by_period
from src.signals.volatility_signal import (
    conviction_size,
    hysteresis_direction,
    signal_statistics,
    vol_target_leverage,
    volatility_score,
)
from src.utils.io import load_config, load_dataframe

QUANTILE_COLUMNS = ["gb_q10", "gb_q90"]
TRUTH_COLUMN = "y_true_rv"

st.set_page_config(page_title="vol_ml_fund", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_pipeline_outputs() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load config and the CSVs produced by the pipeline."""
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    predictions = load_dataframe(processed_dir / "predictions_rv.csv")
    features = load_dataframe(processed_dir / "features.csv")
    prices = load_dataframe(processed_dir / "prices.csv")
    return config, predictions, features, prices


def leg_simple_returns(prices: pd.DataFrame, bt_cfg: dict) -> pd.DataFrame:
    """Simple returns of the two tradable legs, columns long/short."""
    legs = prices[[bt_cfg["long_vol_ticker"], bt_cfg["short_vol_ticker"]]]
    returns = legs.pct_change()
    returns.columns = ["long", "short"]
    return returns


# ---------------------------------------------------------------------------
# Signal + backtest (same building blocks as scripts/run_backtest.py)
# ---------------------------------------------------------------------------

def compute_strategy(
    predictions: pd.DataFrame,
    implied_vol: pd.Series,
    leg_returns: pd.DataFrame,
    model_name: str,
    entry_threshold: float,
    exit_threshold: float,
    sizing_scale: float,
    use_quantile_conviction: bool,
    vol_target: float,
    proxy_vol_window: int,
    max_leverage: float,
    cost_bps: float,
    initial_capital: float,
) -> tuple[BacktestResult, pd.Series, dict[str, float], dict[str, float]]:
    """Score -> hysteresis -> sizing -> vol targeting -> backtest."""
    score = volatility_score(predictions[model_name], implied_vol)
    score_q10 = score_q90 = None
    if use_quantile_conviction:
        score_q10 = volatility_score(predictions["gb_q10"], implied_vol)
        score_q90 = volatility_score(predictions["gb_q90"], implied_vol)

    direction = hysteresis_direction(score, entry_threshold, exit_threshold)
    position = conviction_size(score, direction, sizing_scale, score_q10, score_q90)
    position = vol_target_leverage(
        position, leg_returns, vol_target=vol_target,
        vol_window=proxy_vol_window, max_leverage=max_leverage,
    )
    result = run_backtest(
        weights=position_to_leg_weights(position),
        leg_returns=leg_returns,
        transaction_cost_bps=cost_bps,
        initial_capital=initial_capital,
    )
    summary = performance_summary(result.net_returns, result.equity_curve)
    stats = signal_statistics(result.positions)
    return result, score, summary, stats


@st.cache_data
def sharpe_heatmap(
    model_name: str,
    sizing_scale: float,
    use_quantile_conviction: bool,
    vol_target: float,
    proxy_vol_window: int,
    max_leverage: float,
    cost_bps: float,
) -> pd.DataFrame:
    """Sharpe ratio over an entry x exit threshold grid (exit < entry)."""
    config, predictions, features, prices = load_pipeline_outputs()
    leg_returns = leg_simple_returns(prices, config["backtest"])
    implied_vol = features["implied_vol"]

    entries = np.round(np.arange(0.02, 0.32, 0.04), 2)
    exits = np.round(np.arange(0.00, 0.22, 0.03), 2)
    grid = pd.DataFrame(index=exits, columns=entries, dtype=float)
    grid.index.name = "exit \\ entry"
    for entry in entries:
        for exit_ in exits:
            if exit_ >= entry:
                continue
            _, _, summary, _ = compute_strategy(
                predictions, implied_vol, leg_returns, model_name,
                float(entry), float(exit_), sizing_scale,
                use_quantile_conviction, vol_target, proxy_vol_window,
                max_leverage, cost_bps, 100_000.0,
            )
            grid.loc[exit_, entry] = summary["sharpe_ratio"]
    return grid


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        config, predictions, features, prices = load_pipeline_outputs()
    except FileNotFoundError as error:
        st.error(
            "Sorties du pipeline introuvables. Lancez d'abord :\n\n"
            "`python scripts/run_download.py && python scripts/run_features.py "
            "&& python scripts/run_train.py`\n\n"
            f"Détail : {error}"
        )
        return

    signal_cfg = config["signal"]
    bt_cfg = config["backtest"]
    implied_vol = features["implied_vol"]
    leg_returns = leg_simple_returns(prices, bt_cfg)
    point_models = [
        c for c in predictions.columns
        if c not in QUANTILE_COLUMNS + [TRUTH_COLUMN]
    ]

    st.title("vol_ml_fund — analyse de sensibilité du signal")
    st.caption(
        "Outil de recherche : mêmes modules `src/` que les scripts, aucune "
        "exécution d'ordres. Les prédictions walk-forward sont figées ; seuls "
        "les paramètres du signal et du backtest sont recalculés."
    )

    # ------------------------------------------------------------------ sidebar
    st.sidebar.header("Paramètres du signal")
    default_model = (
        signal_cfg["model_name"] if signal_cfg["model_name"] in point_models
        else point_models[0]
    )
    model_name = st.sidebar.selectbox(
        "Modèle", point_models, index=point_models.index(default_model)
    )
    entry_threshold = st.sidebar.slider(
        "Seuil d'entrée |score|", 0.0, 0.5, float(signal_cfg["entry_threshold"]), 0.01
    )
    exit_threshold = st.sidebar.slider(
        "Seuil de sortie |score|", 0.0, 0.5, float(signal_cfg["exit_threshold"]), 0.01
    )
    if exit_threshold > entry_threshold:
        st.sidebar.warning("Seuil de sortie plafonné au seuil d'entrée.")
        exit_threshold = entry_threshold
    sizing_scale = st.sidebar.slider(
        "Échelle de sizing (score pour taille max)",
        0.05, 1.0, float(signal_cfg["sizing_scale"]), 0.05,
    )
    use_quantile = st.sidebar.checkbox(
        "Réduire la taille si la bande q10-q90 chevauche 0",
        value=bool(signal_cfg["use_quantile_conviction"]),
    )
    st.sidebar.header("Risque et coûts")
    vol_target = st.sidebar.slider(
        "Vol target annualisée", 0.05, 1.0, float(signal_cfg["vol_target"]), 0.05
    )
    max_leverage = st.sidebar.slider(
        "Levier max", 0.25, 2.0, float(signal_cfg["max_leverage"]), 0.25
    )
    cost_bps = st.sidebar.slider(
        "Coûts (bps, aller simple)", 0.0, 50.0,
        float(bt_cfg["transaction_cost_bps"]), 1.0,
    )

    result, score, summary, stats = compute_strategy(
        predictions, implied_vol, leg_returns, model_name,
        entry_threshold, exit_threshold, sizing_scale, use_quantile,
        vol_target, int(signal_cfg["proxy_vol_window"]), max_leverage,
        cost_bps, float(bt_cfg["initial_capital"]),
    )

    tab_backtest, tab_models, tab_market = st.tabs(
        ["Backtest interactif", "Comparaison des modèles", "État du marché"]
    )

    # ------------------------------------------------------------- tab backtest
    with tab_backtest:
        cols = st.columns(5)
        cols[0].metric("Sharpe", f"{summary['sharpe_ratio']:.2f}")
        cols[1].metric("CAGR", f"{summary['cagr']:.1%}")
        cols[2].metric("Max drawdown", f"{summary['max_drawdown']:.1%}")
        cols[3].metric("Vol annualisée", f"{summary['annualized_volatility']:.1%}")
        cols[4].metric("Rendement total", f"{summary['total_return']:.1%}")

        st.subheader("Equity curve (nette de coûts)")
        st.line_chart(result.equity_curve, height=300)

        col_dd, col_pos = st.columns(2)
        with col_dd:
            st.subheader("Drawdown")
            drawdown = result.equity_curve / result.equity_curve.cummax() - 1.0
            st.area_chart(drawdown, height=220)
        with col_pos:
            st.subheader("Position nette (>0 long vol, <0 short vol)")
            st.area_chart(result.positions, height=220)

        st.caption(
            f"Exposition : long {stats['long_share']:.0%} / "
            f"short {stats['short_share']:.0%} / flat {stats['flat_share']:.0%} — "
            f"taille moyenne {stats['avg_abs_position']:.2f}, "
            f"turnover quotidien {stats['avg_daily_turnover']:.3f}, "
            f"coûts cumulés {result.costs.sum():.2%} de drags quotidiens."
        )

        st.subheader("Sensibilité : Sharpe selon les seuils d'entrée / sortie")
        st.caption(
            "Un plateau de Sharpe stable autour de vos seuils est rassurant ; "
            "un pic isolé signale de l'overfitting de seuils (data snooping)."
        )
        if st.button("Calculer la heatmap"):
            grid = sharpe_heatmap(
                model_name, sizing_scale, use_quantile, vol_target,
                int(signal_cfg["proxy_vol_window"]), max_leverage, cost_bps,
            )
            fig, ax = plt.subplots(figsize=(9, 4))
            data = grid.to_numpy(dtype=float)
            image = ax.imshow(data, aspect="auto", cmap="RdYlGn", origin="lower")
            ax.set_xticks(range(len(grid.columns)), [f"{c:.2f}" for c in grid.columns])
            ax.set_yticks(range(len(grid.index)), [f"{i:.2f}" for i in grid.index])
            ax.set_xlabel("Seuil d'entrée")
            ax.set_ylabel("Seuil de sortie")
            for (row, col), value in np.ndenumerate(data):
                if not np.isnan(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                            fontsize=8)
            fig.colorbar(image, ax=ax, label="Sharpe")
            st.pyplot(fig)

    # --------------------------------------------------------------- tab models
    with tab_models:
        y_true = predictions[TRUTH_COLUMN]
        point_forecasts = predictions[point_models]

        st.subheader("Métriques out-of-sample (espace RV)")
        st.dataframe(
            evaluation_table(y_true, point_forecasts).round(4),
            width="stretch",
        )

        col_year, col_dm = st.columns(2)
        with col_year:
            st.subheader("RMSE par année")
            st.dataframe(
                rmse_by_period(y_true, point_forecasts).round(4)
                .style.highlight_min(axis=1, color="#c6efce"),
                width="stretch",
            )
        with col_dm:
            st.subheader("Diebold-Mariano vs HAR")
            st.caption("dm_stat < 0 : le modèle bat le HAR ; p_value : significativité.")
            st.dataframe(
                dm_table(y_true, point_forecasts, benchmark="har",
                         horizon=config["features"]["target_horizon"]).round(4),
                width="stretch",
            )

        st.subheader("Prédictions vs volatilité réalisée")
        years = sorted(predictions.index.year.unique())
        year_start, year_end = st.select_slider(
            "Période", options=years, value=(years[0], years[-1])
        )
        shown_models = st.multiselect(
            "Modèles affichés", point_models,
            default=[m for m in ("ensemble", "random_forest", "har") if m in point_models],
        )
        mask = (predictions.index.year >= year_start) & (
            predictions.index.year <= year_end
        )
        st.line_chart(
            predictions.loc[mask, shown_models + [TRUTH_COLUMN]], height=320
        )

    # --------------------------------------------------------------- tab market
    with tab_market:
        last_date = score.index[-1]
        last_score = float(score.iloc[-1])
        last_position = float(result.positions.iloc[-1])
        st.caption(
            f"Dernière prédiction disponible : {last_date.date()} — la target "
            "nécessite 5 jours de futur, les tout derniers jours n'ont donc "
            "pas encore de prédiction."
        )
        cols = st.columns(4)
        cols[0].metric("Score log(RV prédite / IV)", f"{last_score:+.3f}")
        cols[1].metric(
            "Vol implicite (VIX)", f"{float(implied_vol.loc[last_date]):.1%}"
        )
        cols[2].metric(
            "RV prédite 5j",
            f"{float(predictions[model_name].loc[last_date]):.1%}",
        )
        stance = "LONG vol" if last_position > 0 else (
            "SHORT vol" if last_position < 0 else "FLAT"
        )
        cols[3].metric("Position (après sizing)", f"{last_position:+.2f}", stance)

        st.subheader("Score et seuils — 250 derniers jours")
        recent = score.tail(250).to_frame("score")
        recent["entrée +"] = entry_threshold
        recent["entrée -"] = -entry_threshold
        st.line_chart(recent, height=280)

        st.subheader("RV prédite vs vol implicite — 250 derniers jours")
        comparison = pd.concat(
            [predictions[model_name].rename("RV prédite"),
             implied_vol.reindex(predictions.index).rename("Vol implicite")],
            axis=1,
        ).tail(250)
        st.line_chart(comparison, height=280)


main()

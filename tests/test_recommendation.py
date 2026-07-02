"""Tests for the daily recommendation rule (three-layer architecture)."""

from __future__ import annotations

from datetime import date

import pytest

from src.signals.recommendation import format_message, recommend_position


def test_short_vol_when_contango_and_no_model_warning() -> None:
    reco = recommend_position(
        score=-0.20, term_structure=0.92, short_leg_vol=0.60,
        vol_target=0.30, max_leverage=1.0,
    )
    assert reco.stance == "SHORT_VOL"
    assert reco.raw_position == -1.0
    assert reco.scaled_position == pytest.approx(-0.5)  # 0.30 / 0.60


def test_flat_on_backwardation_even_without_model_warning() -> None:
    reco = recommend_position(score=-0.20, term_structure=1.05, short_leg_vol=0.60)
    assert reco.stance == "FLAT"
    assert reco.scaled_position == 0.0
    assert any("Backwardation" in reason for reason in reco.reasons)


def test_flat_on_model_warning_even_in_contango() -> None:
    reco = recommend_position(score=0.15, term_structure=0.90, short_leg_vol=0.60)
    assert reco.stance == "FLAT"
    assert any("Alerte modèle" in reason for reason in reco.reasons)


def test_leverage_capped_when_proxy_is_quiet() -> None:
    reco = recommend_position(
        score=-0.20, term_structure=0.90, short_leg_vol=0.10,
        vol_target=0.30, max_leverage=1.0,
    )
    assert reco.scaled_position == pytest.approx(-1.0)  # capped, not -3.0


def test_message_contains_recommendation_and_disclaimer() -> None:
    reco = recommend_position(score=-0.20, term_structure=0.92, short_leg_vol=0.60)
    metrics = {
        "vix": 15.0, "vix3m": 16.3, "term_structure": 0.92,
        "predicted_rv": 0.12, "implied_vol": 0.15, "score": -0.22,
        "short_leg_ticker": "SVXY",
    }
    message = format_message(date(2026, 7, 2), metrics, reco)
    assert "SHORT_VOL" in message
    assert "SVXY" in message
    assert "aucune exécution automatique" in message

    stale = format_message(date(2026, 7, 2), metrics, reco, "données en retard")
    assert "données en retard" in stale

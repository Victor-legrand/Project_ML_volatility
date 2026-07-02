"""Daily position recommendation — the three-layer architecture.

Distills the strategy-benchmark findings into one pure, testable rule:

1. **Carry engine**: the base position is short volatility.
2. **Contango filter**: flat when the VIX term structure is in
   backwardation (protection *during* stress regimes).
3. **ML kill switch**: flat when the model's score
   ``log(predicted RV / implied vol)`` warns that volatility is
   underpriced (day selection).

The surviving position is scaled by volatility targeting. This module
computes and formats a recommendation; it never places orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Recommendation:
    """A daily recommendation with its reasoning trail."""

    stance: str                      # "SHORT_VOL" or "FLAT"
    raw_position: float              # -1.0 or 0.0
    scaled_position: float           # after volatility targeting
    reasons: list[str] = field(default_factory=list)


def recommend_position(
    score: float,
    term_structure: float,
    short_leg_vol: float,
    score_threshold: float = 0.0,
    contango_threshold: float = 1.0,
    vol_target: float = 0.30,
    max_leverage: float = 1.0,
) -> Recommendation:
    """Apply the three layers to today's inputs.

    Parameters
    ----------
    score:
        ``log(predicted RV / implied vol)`` from the level model.
    term_structure:
        VIX / VIX3M ratio (>= 1 means backwardation).
    short_leg_vol:
        Annualized rolling volatility of the short-vol proxy (e.g.
        SVXY), used for volatility targeting.
    """
    reasons: list[str] = []
    short_vol = True

    if term_structure >= contango_threshold:
        short_vol = False
        reasons.append(
            f"Backwardation (VIX/VIX3M = {term_structure:.3f} >= "
            f"{contango_threshold:.2f}) : régime de stress, carry coupé."
        )
    else:
        reasons.append(
            f"Contango (VIX/VIX3M = {term_structure:.3f}) : régime favorable au carry."
        )

    if score > score_threshold:
        short_vol = False
        reasons.append(
            f"Alerte modèle (score = {score:+.3f} > {score_threshold:.2f}) : "
            "vol sous-évaluée par le marché, carry coupé."
        )
    else:
        reasons.append(
            f"Pas d'alerte modèle (score = {score:+.3f}) : la vol implicite "
            "couvre la vol prédite."
        )

    if not short_vol:
        return Recommendation("FLAT", 0.0, 0.0, reasons)

    leverage = min(vol_target / short_leg_vol, max_leverage) if short_leg_vol > 0 else 0.0
    reasons.append(
        f"Vol targeting : cible {vol_target:.0%} / vol proxy {short_leg_vol:.0%} "
        f"=> taille {leverage:.2f} (plafond {max_leverage:.2f})."
    )
    return Recommendation("SHORT_VOL", -1.0, -leverage, reasons)


def format_message(
    signal_date: date,
    metrics: dict[str, float],
    recommendation: Recommendation,
    stale_warning: str | None = None,
) -> str:
    """Human-readable daily report (plain text, notification-friendly)."""
    lines = [
        f"vol_ml_fund — signal du {signal_date.isoformat()}",
        "=" * 44,
    ]
    if stale_warning:
        lines += [f"⚠️  {stale_warning}", ""]
    lines += [
        f"VIX             : {metrics['vix']:.2f}",
        f"VIX3M           : {metrics['vix3m']:.2f}",
        f"VIX/VIX3M       : {metrics['term_structure']:.3f}",
        f"RV prédite (5j) : {metrics['predicted_rv']:.1%}",
        f"Vol implicite   : {metrics['implied_vol']:.1%}",
        f"Score           : {metrics['score']:+.3f}",
        "",
        f"RECOMMANDATION  : {recommendation.stance}"
        + (f" (position {recommendation.scaled_position:+.2f} sur "
           f"{metrics.get('short_leg_ticker', 'proxy short-vol')})"
           if recommendation.stance == "SHORT_VOL" else ""),
        "",
        "Raisonnement :",
    ]
    lines += [f"  - {reason}" for reason in recommendation.reasons]
    lines += [
        "",
        "Outil de recherche — aucune exécution automatique. Décision et",
        "ordres restent manuels.",
    ]
    return "\n".join(lines)

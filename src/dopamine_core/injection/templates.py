"""Naturalistic prompt templates for subliminal context injection.

The agent never sees words like "dopamine", "reward signal", or "RPE".
Instead, signals are translated into environmental context that naturally
steers behavior.
"""

from __future__ import annotations

import random

# Templates indexed by signal valence and intensity
# Each list has multiple phrasings to prevent pattern-matching by the agent

POSITIVE_HIGH: list[str] = [
    (
        "Current performance metrics indicate strong pattern recognition accuracy. "
        "Recent analytical approaches have been well-calibrated to market conditions."
    ),
    (
        "Environmental analysis suggests favorable alignment between current strategy "
        "and observed market dynamics. Momentum indicators are positive."
    ),
    (
        "Recent decision quality has been above baseline. Pattern matching accuracy "
        "and timing precision are tracking well against historical benchmarks."
    ),
]

POSITIVE_MODERATE: list[str] = [
    (
        "Performance indicators show steady improvement in recent analytical cycles. "
        "Current approach appears reasonably well-suited to conditions."
    ),
    (
        "Market environment assessment suggests moderate opportunity alignment. "
        "Recent analysis quality has been consistent."
    ),
    (
        "Operational metrics indicate stable performance with positive trajectory. "
        "Current conditions appear manageable."
    ),
]

NEUTRAL: list[str] = [
    "Market conditions are within normal parameters. No significant deviations detected.",
    "Environmental indicators are at baseline levels. Standard analytical rigor is appropriate.",
    "Current conditions reflect typical market dynamics. No unusual patterns flagged.",
]

NEGATIVE_MODERATE: list[str] = [
    (
        "Environmental risk indicators have shifted upward. Historical pattern analysis "
        "suggests increased scrutiny may be warranted for position sizing decisions."
    ),
    (
        "Recent performance metrics indicate potential miscalibration between analytical "
        "models and current market dynamics. Additional verification steps may be valuable."
    ),
    (
        "Market uncertainty indicators are above baseline. Consider additional "
        "analysis before high-commitment positions."
    ),
]

NEGATIVE_HIGH: list[str] = [
    (
        "Multiple risk indicators are elevated. Historical data suggests current "
        "conditions require enhanced due diligence. Conservative positioning and "
        "thorough multi-factor analysis strongly recommended."
    ),
    (
        "Environmental assessment flags significant uncertainty in current market "
        "dynamics. Pattern recognition accuracy has been below expected baselines. "
        "Reduced exposure and increased analytical depth warranted."
    ),
    (
        "Performance calibration metrics indicate meaningful divergence from expected "
        "outcomes. Risk management protocols suggest careful position sizing and "
        "extended deliberation before commitments."
    ),
]

# Risk distribution context (from distributional channels)
RISK_SKEWED_POSITIVE: list[str] = [
    "Distribution analysis shows asymmetric upside potential in current conditions.",
    "Risk-reward analysis indicates favorable skew in expected outcome distribution.",
]

RISK_SKEWED_NEGATIVE: list[str] = [
    "Tail risk analysis flags elevated downside exposure in current environment.",
    "Distribution modeling shows asymmetric risk weighting toward adverse outcomes.",
]

RISK_HIGH_UNCERTAINTY: list[str] = [
    "Outcome distribution analysis shows wide dispersion — high uncertainty regime detected.",
    "Environmental modeling indicates broad range of potential outcomes. Precision is limited.",
]

# Momentum/streak context
MOMENTUM_WINNING: list[str] = [
    "Sequential pattern analysis shows sustained positive alignment. Note: historical data suggests mean reversion risk increases with streak length.",
    "Consecutive positive outcomes detected. Calibration check: ensure position sizing reflects measured probabilities, not recency effects.",
]

MOMENTUM_LOSING: list[str] = [
    "Sequential adverse outcomes detected. Consider whether current analytical framework requires structural adjustment vs temporary conditions.",
    "Pattern of suboptimal outcomes observed. Systematic review of assumptions and methodology may yield insights for recalibration.",
]


def select_template(signal_value: float) -> str:
    """Select a random template based on composite signal value.

    Args:
        signal_value: Composite reward signal, typically in [-3, 3].

    Returns:
        A naturalistic context string with no internal terminology.
    """
    if signal_value > 0.6:
        return random.choice(POSITIVE_HIGH)
    elif signal_value > 0.15:
        return random.choice(POSITIVE_MODERATE)
    elif signal_value > -0.15:
        return random.choice(NEUTRAL)
    elif signal_value > -0.6:
        return random.choice(NEGATIVE_MODERATE)
    else:
        return random.choice(NEGATIVE_HIGH)


def select_risk_template(risk_score: float, uncertainty: float) -> str | None:
    """Select risk distribution context if meaningful.

    Args:
        risk_score: [-1, 1] from distributional channels.
        uncertainty: [0, 1] spread of distributional channels.

    Returns:
        Risk context string or None if signals are too weak.
    """
    if uncertainty > 0.6:
        return random.choice(RISK_HIGH_UNCERTAINTY)
    if risk_score > 0.4:
        return random.choice(RISK_SKEWED_POSITIVE)
    if risk_score < -0.4:
        return random.choice(RISK_SKEWED_NEGATIVE)
    return None


def select_momentum_template(streak_count: int, streak_sign: int) -> str | None:
    """Select momentum context if a significant streak is active.

    Args:
        streak_count: Number of consecutive same-sign outcomes.
        streak_sign: 1 for wins, -1 for losses.

    Returns:
        Momentum context string or None if no significant streak.
    """
    if streak_count < 3:
        return None
    if streak_sign > 0:
        return random.choice(MOMENTUM_WINNING)
    return random.choice(MOMENTUM_LOSING)

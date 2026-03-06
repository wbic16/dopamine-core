"""Momentum tracking — win/loss streak detection and momentum signals.

Tracks consecutive positive or negative outcomes and produces a momentum
factor that influences the composite reward signal. Sustained streaks
amplify the signal, while streak breaks reset momentum.
"""

from __future__ import annotations

from dopamine_core.config import MomentumConfig


class MomentumTracker:
    """Tracks win/loss streaks and computes momentum factors."""

    def __init__(self, config: MomentumConfig | None = None) -> None:
        self._config = config or MomentumConfig()
        self._streak_count: int = 0
        self._streak_sign: int = 0  # 1 = wins, -1 = losses, 0 = none
        self._cooldown_remaining: int = 0

    @property
    def streak_count(self) -> int:
        return self._streak_count

    @property
    def streak_sign(self) -> int:
        return self._streak_sign

    def update(self, pnl: float) -> None:
        """Update streak state with new trade outcome.

        Args:
            pnl: Raw PnL from the trade.
        """
        current_sign = 1 if pnl > 0 else (-1 if pnl < 0 else 0)

        if current_sign == 0:
            return

        if current_sign == self._streak_sign:
            self._streak_count += 1
            self._cooldown_remaining = 0
        else:
            # Streak broken — start cooldown
            if self._streak_count >= self._config.streak_threshold:
                self._cooldown_remaining = self._config.cooldown_steps
            self._streak_sign = current_sign
            self._streak_count = 1

    def get_momentum_factor(self) -> float:
        """Compute the current momentum factor.

        Returns:
            Signed momentum factor. Positive for win streaks,
            negative for loss streaks, 0 if below threshold or cooling down.
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return 0.0

        cfg = self._config
        if self._streak_count < cfg.streak_threshold:
            return 0.0

        excess = self._streak_count - cfg.streak_threshold
        factor = min(1.0 + excess * 0.1, cfg.max_streak_multiplier)
        return factor * self._streak_sign

    def reset(self) -> None:
        self._streak_count = 0
        self._streak_sign = 0
        self._cooldown_remaining = 0

    def load(self, streak_count: int, streak_sign: int) -> None:
        self._streak_count = streak_count
        self._streak_sign = streak_sign
        self._cooldown_remaining = 0

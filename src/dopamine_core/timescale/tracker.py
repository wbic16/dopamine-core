"""Multi-timescale integration — tracks reward signals at different temporal granularities.

Biological analogy: dopamine operates on multiple timescales simultaneously.
Fast signals (milliseconds/token-level) capture immediate surprises, while
slow signals (minutes/session-level) track overall performance trends.

The system maintains separate EMA trackers for each timescale level and
produces a weighted composite that balances short-term reactivity with
long-term stability.
"""

from __future__ import annotations

from dopamine_core.config import TimescaleConfig
from dopamine_core.types import TimescaleLevel


class TimescaleTracker:
    """Tracks reward signals across multiple timescales.

    Each timescale has its own EMA with a different smoothing factor:
    - Token: very fast (alpha=0.5), captures immediate signal changes
    - Step: fast (alpha=0.2), tracks per-trade dynamics
    - Episode: medium (alpha=0.05), tracks session segments
    - Session: slow (alpha=0.01), tracks overall trends
    """

    # EMA smoothing factors per timescale
    _ALPHA: dict[TimescaleLevel, float] = {
        TimescaleLevel.TOKEN: 0.5,
        TimescaleLevel.STEP: 0.2,
        TimescaleLevel.EPISODE: 0.05,
        TimescaleLevel.SESSION: 0.01,
    }

    def __init__(self, config: TimescaleConfig | None = None) -> None:
        self._config = config or TimescaleConfig()
        self._levels: dict[TimescaleLevel, float] = {
            level: 0.0 for level in TimescaleLevel
        }
        self._step_counts: dict[TimescaleLevel, int] = {
            level: 0 for level in TimescaleLevel
        }

    def update(self, value: float, level: TimescaleLevel = TimescaleLevel.STEP) -> None:
        """Update a specific timescale level with a new signal value.

        Also propagates to slower timescales: updating STEP also updates
        EPISODE and SESSION with the same value.

        Args:
            value: The signal value to integrate.
            level: The timescale level to update (and all slower ones).
        """
        levels_ordered = [
            TimescaleLevel.TOKEN,
            TimescaleLevel.STEP,
            TimescaleLevel.EPISODE,
            TimescaleLevel.SESSION,
        ]

        start_idx = levels_ordered.index(level)
        for lvl in levels_ordered[start_idx:]:
            alpha = self._ALPHA[lvl]
            self._levels[lvl] = self._levels[lvl] * (1.0 - alpha) + alpha * value
            self._step_counts[lvl] += 1

    def get_level(self, level: TimescaleLevel) -> float:
        """Get the current EMA value at a specific timescale."""
        return self._levels[level]

    def get_composite(self) -> float:
        """Compute weighted composite across all timescales.

        Uses the configured weights to blend all timescale levels
        into a single value.

        Returns:
            Weighted composite signal.
        """
        weights = {
            TimescaleLevel.TOKEN: self._config.token_weight,
            TimescaleLevel.STEP: self._config.step_weight,
            TimescaleLevel.EPISODE: self._config.episode_weight,
            TimescaleLevel.SESSION: self._config.session_weight,
        }

        total_weight = sum(weights.values())
        if total_weight < 1e-9:
            return 0.0

        composite = sum(
            weights[lvl] * self._levels[lvl] for lvl in TimescaleLevel
        )
        return composite / total_weight

    def get_divergence(self) -> float:
        """Measure divergence between fast and slow timescales.

        Large divergence indicates a regime change — fast timescales
        have moved but slow ones haven't caught up.

        Returns:
            Divergence value (fast EMA - slow EMA).
        """
        fast = self._levels[TimescaleLevel.STEP]
        slow = self._levels[TimescaleLevel.SESSION]
        return fast - slow

    def reset(self) -> None:
        for level in TimescaleLevel:
            self._levels[level] = 0.0
            self._step_counts[level] = 0

    def get_state(self) -> dict[str, float]:
        return {level.value: self._levels[level] for level in TimescaleLevel}

    def load_state(self, state: dict[str, float]) -> None:
        for level in TimescaleLevel:
            if level.value in state:
                self._levels[level] = state[level.value]

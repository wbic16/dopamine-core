"""Phasic reward processor — event-specific bursts and dips.

Biological analogy: phasic dopamine firing represents the immediate
response to a surprising outcome. Bursts (positive RPE) reinforce
the preceding behavior; dips (negative RPE) suppress it.

Phasic signals decay over subsequent steps, modeling the transient
nature of event-specific dopamine responses.
"""

from __future__ import annotations

import math
from collections import deque

from dopamine_core.config import PhasicConfig


class PhasicProcessor:
    """Processes and tracks phasic (event-specific) reward signals.

    Maintains a history of recent phasic bursts/dips with exponential decay,
    so the system remembers recent events with diminishing influence.
    """

    def __init__(self, config: PhasicConfig | None = None) -> None:
        self._config = config or PhasicConfig()
        # (signal_value, steps_ago) — most recent first
        self._history: deque[tuple[float, int]] = deque(maxlen=20)
        self._current_response: float = 0.0

    @property
    def current_response(self) -> float:
        return self._current_response

    @property
    def pnl_scale(self) -> float:
        return self._config.pnl_scale

    def process(self, rpe_error: float, raw_error: float) -> float:
        """Process a new RPE into a scaled phasic signal.

        Args:
            rpe_error: Loss-aversion-adjusted RPE error.
            raw_error: Raw RPE before loss aversion.

        Returns:
            Scaled phasic response value.
        """
        # Apply asymmetric scaling: bursts for positive, dips for negative
        if raw_error >= 0:
            scaled = rpe_error * self._config.burst_scale
        else:
            scaled = rpe_error * self._config.dip_scale

        self._current_response = scaled

        # Age existing history entries
        aged: deque[tuple[float, int]] = deque(maxlen=20)
        for value, age in self._history:
            aged.append((value, age + 1))
        self._history = aged

        # Add new signal
        self._history.appendleft((scaled, 0))

        return scaled

    def get_decayed_influence(self) -> float:
        """Get the cumulative influence of recent phasic signals with decay.

        Returns:
            Weighted sum of recent phasic signals, decayed by age.
        """
        if not self._history:
            return 0.0

        half_life = self._config.decay_half_life
        total = 0.0
        for value, age in self._history:
            decay = math.exp(-0.693 * age / half_life)  # ln(2) ≈ 0.693
            total += value * decay

        return total

    def reset(self) -> None:
        self._history.clear()
        self._current_response = 0.0

    def get_history(self) -> list[tuple[float, int]]:
        return list(self._history)

    def load_history(self, history: list[tuple[float, int]]) -> None:
        self._history = deque(history, maxlen=20)
        if history:
            self._current_response = history[0][0]

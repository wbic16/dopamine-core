"""Tonic baseline — slow-adapting expectation level.

Biological analogy: tonic dopamine firing rate reflects the general
motivational state. It rises during sustained success and falls during
sustained failure, creating an adaptive reference point for RPE computation.
"""

from __future__ import annotations

from dopamine_core.config import TonicConfig


class TonicBaseline:
    """Manages the slow-adapting tonic baseline.

    The baseline moves via exponential moving average of RPE signals,
    with configurable learning rate and decay.
    """

    def __init__(self, config: TonicConfig | None = None) -> None:
        self._config = config or TonicConfig()
        self._baseline: float = self._config.initial_baseline
        self._step_count: int = 0

    @property
    def level(self) -> float:
        return self._baseline

    @property
    def step_count(self) -> int:
        return self._step_count

    def update(self, rpe_signal: float) -> float:
        """Update baseline with new RPE signal.

        Args:
            rpe_signal: Raw RPE value (before loss aversion).

        Returns:
            Updated baseline level.
        """
        cfg = self._config
        self._baseline = (
            self._baseline * cfg.decay_rate
            + cfg.learning_rate * (rpe_signal - self._baseline)
        )
        self._baseline = max(-cfg.max_level, min(cfg.max_level, self._baseline))
        self._step_count += 1
        return self._baseline

    def reset(self) -> None:
        self._baseline = self._config.initial_baseline
        self._step_count = 0

    def load(self, baseline: float, step_count: int) -> None:
        self._baseline = baseline
        self._step_count = step_count

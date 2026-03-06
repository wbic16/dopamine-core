"""Dual-mode reward integration — combines tonic baseline with phasic responses.

The composite signal blends the slow-adapting tonic level (general motivation)
with the event-specific phasic response (immediate reaction), producing
a single value that drives context injection.
"""

from __future__ import annotations

from dopamine_core.config import PhasicConfig, TonicConfig
from dopamine_core.reward.phasic import PhasicProcessor
from dopamine_core.reward.tonic import TonicBaseline
from dopamine_core.types import RPEResult


class DualModeReward:
    """Integrates tonic and phasic reward channels into a composite signal.

    The composite value is a weighted blend:
        composite = tonic_weight * tonic_level + phasic_weight * phasic_response

    This ensures both sustained motivational state and immediate event
    reactions contribute to the agent's behavioral steering.
    """

    def __init__(
        self,
        tonic_config: TonicConfig | None = None,
        phasic_config: PhasicConfig | None = None,
        tonic_weight: float = 0.3,
        phasic_weight: float = 0.7,
    ) -> None:
        self._tonic = TonicBaseline(tonic_config)
        self._phasic = PhasicProcessor(phasic_config)
        self._tonic_weight = tonic_weight
        self._phasic_weight = phasic_weight

    @property
    def tonic(self) -> TonicBaseline:
        return self._tonic

    @property
    def phasic(self) -> PhasicProcessor:
        return self._phasic

    @property
    def pnl_scale(self) -> float:
        return self._phasic.pnl_scale

    def process(self, rpe: RPEResult) -> float:
        """Process an RPE result through both tonic and phasic channels.

        Args:
            rpe: The computed RPE result from RPECalculator.

        Returns:
            Composite reward signal value.
        """
        # Update tonic baseline with raw RPE (before loss aversion)
        self._tonic.update(rpe.raw_error)

        # Process phasic response with full RPE (including loss aversion)
        phasic_value = self._phasic.process(rpe.error, rpe.raw_error)

        # Blend tonic and phasic
        composite = (
            self._tonic_weight * self._tonic.level
            + self._phasic_weight * phasic_value
        )
        return composite

    def reset(self) -> None:
        self._tonic.reset()
        self._phasic.reset()

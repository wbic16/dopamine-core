"""Distributional coding — translates channel state into risk/uncertainty signals.

Bridges the distributional channel representation with the injection system,
extracting actionable risk and uncertainty metrics from the channel distribution.
"""

from __future__ import annotations

from dopamine_core.distributional.channels import DistributionalChannels


class DistributionalCoding:
    """Extracts risk and uncertainty signals from distributional channels."""

    def __init__(self, channels: DistributionalChannels) -> None:
        self._channels = channels

    def get_risk_score(self) -> float:
        """Compute a risk score from the distributional state.

        Combines skew and the relative position of pessimistic channels.
        Positive → upside skew (favorable risk/reward).
        Negative → downside skew (unfavorable risk/reward).

        Returns:
            Risk score in [-1, 1].
        """
        skew = self._channels.get_skew()
        exps = self._channels.expectations
        if len(exps) < 2:
            return 0.0

        # Weight by how much pessimistic channels (low tau) diverge
        # from optimistic ones (high tau)
        pessimistic_mean = sum(exps[: len(exps) // 2]) / max(len(exps) // 2, 1)
        optimistic_mean = sum(exps[len(exps) // 2 :]) / max(len(exps) - len(exps) // 2, 1)

        spread = self._channels.get_spread()
        if spread < 1e-9:
            return 0.0

        # Asymmetry between pessimistic and optimistic halves
        asymmetry = (optimistic_mean - pessimistic_mean) / spread if spread > 0 else 0.0
        risk_score = 0.5 * skew + 0.5 * max(-1.0, min(1.0, asymmetry))

        return max(-1.0, min(1.0, risk_score))

    def get_uncertainty(self) -> float:
        """Compute uncertainty from the distributional spread.

        Maps the spread of channel expectations to a [0, 1] value.
        Higher spread → higher uncertainty about reward expectations.

        Returns:
            Uncertainty in [0, 1].
        """
        spread = self._channels.get_spread()
        # Normalize: spread of ~1.0 → uncertainty ~0.5
        # Using a soft mapping to avoid hard cutoffs
        import math

        return min(1.0, math.tanh(spread * 2.0))

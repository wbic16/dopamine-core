"""Multi-channel distributional reward coding.

Biological analogy: different dopamine neurons encode different quantiles
of the reward distribution. Some are "optimistic" (respond to better-than-expected
outcomes), others are "pessimistic" (respond to worse-than-expected). Together
they maintain a distributional representation of reward expectations.

Reference: Dabney et al. (2020) — "A distributional code for value in
dopamine-based reinforcement learning", Nature.
"""

from __future__ import annotations

from dopamine_core.config import DistributionalConfig


class RewardChannel:
    """A single reward channel tracking one quantile of the reward distribution.

    Each channel has a quantile level (tau) that determines its asymmetric
    learning: channels with high tau learn more from positive surprises,
    channels with low tau learn more from negative surprises.
    """

    def __init__(self, tau: float, learning_rate: float) -> None:
        self._tau = tau
        self._learning_rate = learning_rate
        self._expectation: float = 0.0

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def expectation(self) -> float:
        return self._expectation

    def update(self, outcome: float) -> float:
        """Update this channel's expectation using asymmetric quantile learning.

        Args:
            outcome: Normalized outcome value.

        Returns:
            The prediction error for this channel.
        """
        error = outcome - self._expectation

        # Asymmetric learning: tau determines the balance
        # High tau → learns more from positive errors (optimistic)
        # Low tau → learns more from negative errors (pessimistic)
        if error >= 0:
            step = self._learning_rate * self._tau * error
        else:
            step = self._learning_rate * (1.0 - self._tau) * error

        self._expectation += step
        return error

    def load(self, expectation: float) -> None:
        self._expectation = expectation


class DistributionalChannels:
    """Maintains multiple reward channels spanning the quantile spectrum.

    Creates a set of channels with evenly spaced tau values from tau_min
    to tau_max. Together, these channels approximate the full distribution
    of reward expectations.
    """

    def __init__(self, config: DistributionalConfig | None = None) -> None:
        self._config = config or DistributionalConfig()
        self._channels = self._create_channels()

    def _create_channels(self) -> list[RewardChannel]:
        cfg = self._config
        n = cfg.num_channels
        if n == 1:
            taus = [0.5]
        else:
            step = (cfg.tau_max - cfg.tau_min) / (n - 1)
            taus = [cfg.tau_min + i * step for i in range(n)]

        return [RewardChannel(tau=t, learning_rate=cfg.learning_rate) for t in taus]

    @property
    def num_channels(self) -> int:
        return len(self._channels)

    @property
    def expectations(self) -> list[float]:
        return [ch.expectation for ch in self._channels]

    def update(self, outcome: float) -> list[float]:
        """Update all channels with a new outcome.

        Args:
            outcome: Normalized outcome value.

        Returns:
            List of per-channel prediction errors.
        """
        return [ch.update(outcome) for ch in self._channels]

    def get_mean_expectation(self) -> float:
        """Average expectation across all channels."""
        exps = self.expectations
        return sum(exps) / len(exps) if exps else 0.0

    def get_spread(self) -> float:
        """Spread between optimistic and pessimistic channels.

        A large spread indicates high uncertainty about the reward distribution.
        """
        exps = self.expectations
        if len(exps) < 2:
            return 0.0
        return max(exps) - min(exps)

    def get_skew(self) -> float:
        """Asymmetry of the distribution.

        Positive skew → optimistic channels see higher expectations.
        Negative skew → pessimistic channels are more extreme.

        Returns:
            Skew value in [-1, 1].
        """
        exps = self.expectations
        if len(exps) < 3:
            return 0.0

        mean = sum(exps) / len(exps)
        median_idx = len(exps) // 2
        median = exps[median_idx]

        spread = self.get_spread()
        if spread < 1e-9:
            return 0.0

        skew = (mean - median) / spread
        return max(-1.0, min(1.0, skew))

    def reset(self) -> None:
        self._channels = self._create_channels()

    def load_expectations(self, expectations: list[float]) -> None:
        for ch, exp in zip(self._channels, expectations):
            ch.load(exp)

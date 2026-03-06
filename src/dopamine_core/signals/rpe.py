"""Reward Prediction Error computation with confidence weighting and loss aversion."""

from __future__ import annotations

from dopamine_core.config import LossAversionConfig
from dopamine_core.types import RPEResult

_EPSILON = 0.01


class RPECalculator:
    """Computes Reward Prediction Error using the confidence-weighted formula.

    ## Formula (explicit)

    The formula:
        raw_error = outcome * (1 - conf) + (1 - outcome) * (-conf)

    Algebraically simplifies to:
        raw_error = outcome - conf       (confidence is the implicit prediction)

    This treats expressed **confidence as the agent's probability estimate**
    of a positive outcome. RPE is then how much better or worse the actual
    outcome was versus what the agent expected.

    Consequences:
    - High confidence + wrong = large negative signal (overconfidence penalty)
    - Low confidence + right  = large positive signal (pleasant surprise)
    - Low confidence + wrong  = zero signal (expected loss, no update)
    - High confidence + right = zero signal (expected win, no update)

    Note on the `baseline` parameter:
        `baseline` is the current tonic level. It is NOT used in the RPE
        computation itself — confidence serves as the prediction. The baseline
        is used only to compute `surprise` (how large is the error relative
        to the running average). Pass it so that surprise is meaningful;
        it does not affect tonic adaptation.

    Loss aversion amplifies negative RPE by 1.87x (Kahneman & Tversky).
    """

    def __init__(self, config: LossAversionConfig | None = None) -> None:
        self._config = config or LossAversionConfig()

    def compute(
        self,
        outcome: float,
        confidence: float,
        baseline: float = 0.0,
    ) -> RPEResult:
        """Compute confidence-weighted RPE.

        Args:
            outcome: Normalized outcome in [0, 1] where 1 = full win, 0 = full loss.
            confidence: Agent's extracted confidence in [-1, 1].
                        Mapped to [0, 1] for the formula; serves as the implicit prediction.
            baseline: Current tonic baseline (used only for surprise normalization,
                      not for the RPE computation itself).

        Returns:
            RPEResult with raw and loss-aversion-adjusted error.
        """
        # Map confidence from [-1, 1] to [0, 1]
        # This is the implicit prediction: conf_normalized ≈ P(win)
        conf_normalized = (confidence + 1.0) / 2.0

        # RPE = outcome - confidence  (algebraic simplification of the formula)
        # Kept in original form for clarity and consistency with the research paper
        raw_error = outcome * (1.0 - conf_normalized) + (1.0 - outcome) * (-conf_normalized)

        # Apply loss aversion: negative errors (losses vs expectation) are amplified
        if raw_error < 0:
            error = raw_error * self._config.multiplier
        else:
            error = raw_error

        # Surprise = magnitude of error relative to tonic baseline magnitude
        # (baseline not used in RPE, used here to scale the surprise signal)
        surprise = abs(raw_error) / max(abs(baseline), _EPSILON)

        return RPEResult(
            prediction=conf_normalized,   # the implicit prediction (confidence)
            actual=outcome,
            error=error,
            raw_error=raw_error,
            surprise=min(surprise, 10.0),  # cap surprise to avoid extremes
        )

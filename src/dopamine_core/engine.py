"""DopamineEngine — the central orchestrator for synthetic dopamine signals."""

from __future__ import annotations

from collections import deque

from dopamine_core.config import DopamineConfig
from dopamine_core.injection.context import ContextInjector
from dopamine_core.signals.extractor import SignalExtractor
from dopamine_core.signals.rpe import RPECalculator
from dopamine_core.types import (
    CompositeSignal,
    EngineState,
    ExtractedSignals,
    Outcome,
    RPEResult,
)


class DopamineEngine:
    """Main entry point for DopamineCore.

    Usage::

        engine = DopamineEngine()

        # In your agent loop
        prompt = engine.inject_context(base_prompt)
        response = agent.complete(prompt)
        outcome = evaluate_trade(response)
        engine.update(response.text, outcome)
    """

    def __init__(self, config: DopamineConfig | None = None) -> None:
        self._config = config or DopamineConfig()

        # Core components
        self._extractor = SignalExtractor()
        self._rpe_calc = RPECalculator(self._config.loss_aversion)
        self._injector = ContextInjector(self._config.injection)

        # State
        self._tonic_baseline: float = self._config.tonic.initial_baseline
        self._step_count: int = 0
        self._outcome_history: deque[float] = deque(maxlen=100)
        self._streak_count: int = 0
        self._streak_sign: int = 0  # 1 = wins, -1 = losses, 0 = none
        self._last_rpe: float = 0.0
        self._last_signals: ExtractedSignals | None = None
        self._last_composite: CompositeSignal | None = None

    def inject_context(self, base_prompt: str) -> str:
        """Add subliminal reward context to a prompt.

        Uses the current internal state to generate naturalistic
        environmental context that steers agent behavior.

        Args:
            base_prompt: The original prompt to augment.

        Returns:
            Augmented prompt with injected motivation context.
        """
        signal = self._build_current_signal()
        context = self._injector.build_context(signal)
        return self._injector.inject(base_prompt, context)

    def update(
        self,
        response_text: str,
        outcome: Outcome | float,
    ) -> CompositeSignal:
        """Process an outcome and update all internal state.

        Args:
            response_text: The agent's CoT/response text.
            outcome: Trade outcome — either an Outcome object or raw PnL float.

        Returns:
            The computed composite reward signal.
        """
        # Normalize outcome
        if isinstance(outcome, (int, float)):
            outcome = Outcome(pnl=float(outcome))

        # 1. Extract behavioral signals from response
        signals = self._extractor.extract(response_text)
        self._last_signals = signals

        # 2. Normalize PnL to [0, 1] for RPE formula
        normalized_outcome = self._normalize_pnl(outcome.pnl)

        # 3. Compute RPE
        confidence = outcome.confidence if outcome.confidence is not None else signals.confidence
        rpe = self._rpe_calc.compute(
            outcome=normalized_outcome,
            confidence=confidence,
            baseline=self._tonic_baseline,
        )
        self._last_rpe = rpe.error

        # 4. Update tonic baseline (slow EMA) — driven by RPE, not raw outcome
        self._update_tonic(rpe.raw_error)

        # 5. Update streak tracking
        self._update_streak(outcome.pnl)

        # 6. Record outcome
        self._outcome_history.append(outcome.pnl)
        self._step_count += 1

        # 7. Apply safety clamping
        clamped_rpe = self._clamp(rpe.error)

        # 8. Build composite signal
        composite = CompositeSignal(
            value=clamped_rpe,
            confidence_factor=confidence,
            risk_assessment=signals.risk_framing,
            momentum_factor=self._compute_momentum_factor(),
            tonic_level=self._tonic_baseline,
            phasic_response=rpe.error,
        )
        self._last_composite = composite
        return composite

    def get_state(self) -> EngineState:
        """Snapshot full state for serialization/persistence."""
        return EngineState(
            tonic_baseline=self._tonic_baseline,
            step_count=self._step_count,
            outcome_history=list(self._outcome_history),
            streak_count=self._streak_count,
            streak_sign=self._streak_sign,
            last_rpe=self._last_rpe,
        )

    def load_state(self, state: EngineState) -> None:
        """Restore engine from a saved state snapshot."""
        self._tonic_baseline = state.tonic_baseline
        self._step_count = state.step_count
        self._outcome_history = deque(state.outcome_history, maxlen=100)
        self._streak_count = state.streak_count
        self._streak_sign = state.streak_sign
        self._last_rpe = state.last_rpe

    def reset(self) -> None:
        """Reset all state to initial values."""
        self._tonic_baseline = self._config.tonic.initial_baseline
        self._step_count = 0
        self._outcome_history.clear()
        self._streak_count = 0
        self._streak_sign = 0
        self._last_rpe = 0.0
        self._last_signals = None
        self._last_composite = None

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def tonic_baseline(self) -> float:
        return self._tonic_baseline

    @property
    def last_composite(self) -> CompositeSignal | None:
        return self._last_composite

    # --- Private methods ---

    def _build_current_signal(self) -> CompositeSignal:
        """Build a signal from current state (for injection before update)."""
        if self._last_composite is not None:
            return self._last_composite
        return CompositeSignal(
            value=0.0,
            confidence_factor=0.0,
            risk_assessment=0.0,
            momentum_factor=0.0,
            tonic_level=self._tonic_baseline,
            phasic_response=0.0,
        )

    def _normalize_pnl(self, pnl: float) -> float:
        """Normalize raw PnL to [0, 1] using a sigmoid-like mapping."""
        # Simple tanh-based normalization, then shift to [0, 1]
        import math

        normalized = math.tanh(pnl / 100.0)  # scale factor for typical PnL values
        return (normalized + 1.0) / 2.0  # shift from [-1,1] to [0,1]

    def _update_tonic(self, normalized_outcome: float) -> None:
        """Update tonic baseline with exponential moving average."""
        cfg = self._config.tonic
        self._tonic_baseline = (
            self._tonic_baseline * cfg.decay_rate
            + cfg.learning_rate * (normalized_outcome - self._tonic_baseline)
        )
        # Clamp to bounds
        max_level = cfg.max_level
        self._tonic_baseline = max(-max_level, min(max_level, self._tonic_baseline))

    def _update_streak(self, pnl: float) -> None:
        """Update win/loss streak tracking."""
        current_sign = 1 if pnl > 0 else (-1 if pnl < 0 else 0)

        if current_sign == 0:
            return  # neutral outcome doesn't break or extend streak

        if current_sign == self._streak_sign:
            self._streak_count += 1
        else:
            self._streak_sign = current_sign
            self._streak_count = 1

    def _compute_momentum_factor(self) -> float:
        """Compute momentum multiplier from streak state."""
        cfg = self._config.momentum
        if self._streak_count < cfg.streak_threshold:
            return 0.0

        # Scale linearly from threshold to max
        excess = self._streak_count - cfg.streak_threshold
        factor = min(1.0 + excess * 0.1, cfg.max_streak_multiplier)
        return factor * self._streak_sign

    def _clamp(self, value: float) -> float:
        """Clamp signal to safe bounds."""
        mag = self._config.safety.max_signal_magnitude
        return max(-mag, min(mag, value))

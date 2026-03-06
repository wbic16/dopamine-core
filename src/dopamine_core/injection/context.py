"""Subliminal context injection — translates reward signals into naturalistic prompt context."""

from __future__ import annotations

from dopamine_core.config import InjectionConfig
from dopamine_core.injection.templates import (
    select_momentum_template,
    select_risk_template,
    select_template,
)
from dopamine_core.types import CompositeSignal


class ContextInjector:
    """Builds and injects subliminal reward context into agent prompts.

    The agent never sees "dopamine" or "reward" — only naturalistic
    environmental context that steers behavior.
    """

    def __init__(self, config: InjectionConfig | None = None) -> None:
        self._config = config or InjectionConfig()

    def build_context(self, signal: CompositeSignal) -> str:
        """Translate composite reward signal into naturalistic text context.

        Args:
            signal: Aggregated reward signal from the engine.

        Returns:
            Environmental context string ready for injection.
        """
        parts: list[str] = []

        # Main reward-based context
        parts.append(select_template(signal.value))

        # Risk distribution context (if distributional channels provide signal)
        uncertainty = abs(signal.risk_assessment)
        risk_ctx = select_risk_template(signal.risk_assessment, uncertainty)
        if risk_ctx:
            parts.append(risk_ctx)

        # Momentum context (if streak is active)
        streak_count = int(abs(signal.momentum_factor) * 10)  # rough mapping
        streak_sign = 1 if signal.momentum_factor > 0 else -1
        momentum_ctx = select_momentum_template(streak_count, streak_sign)
        if momentum_ctx:
            parts.append(momentum_ctx)

        return " ".join(parts)

    def inject(self, base_prompt: str, context: str) -> str:
        """Insert context into the prompt using the configured style.

        Args:
            base_prompt: The original prompt to augment.
            context: The environmental context to inject.

        Returns:
            Augmented prompt with subliminal context.
        """
        if not context.strip():
            return base_prompt

        if self._config.style == "environmental":
            return (
                f"[Environmental Context]\n{context}\n"
                f"[End Environmental Context]\n\n{base_prompt}"
            )
        elif self._config.style == "system":
            return f"System note: {context}\n\n{base_prompt}"
        else:  # prefix
            return f"{context}\n\n{base_prompt}"

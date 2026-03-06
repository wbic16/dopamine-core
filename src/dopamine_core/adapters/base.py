"""Base adapter interface for framework integrations.

Adapters bridge DopamineCore with popular agent frameworks (LangChain, CrewAI, etc.)
by hooking into the framework's prompt/response lifecycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dopamine_core.config import DopamineConfig
from dopamine_core.engine import DopamineEngine
from dopamine_core.types import CompositeSignal, Outcome


class DopamineAdapter(ABC):
    """Abstract base for framework-specific DopamineCore adapters.

    Subclasses implement framework-specific hooks for:
    1. Intercepting prompts before they reach the LLM (inject context)
    2. Intercepting responses after the LLM returns (extract signals)
    3. Processing outcomes (update reward state)
    """

    def __init__(self, config: DopamineConfig | None = None) -> None:
        self._engine = DopamineEngine(config)

    @property
    def engine(self) -> DopamineEngine:
        return self._engine

    def wrap_prompt(self, prompt: str) -> str:
        """Inject dopamine context into a prompt before sending to LLM.

        Args:
            prompt: The original prompt text.

        Returns:
            Augmented prompt with subliminal reward context.
        """
        return self._engine.inject_context(prompt)

    def process_response(self, response_text: str, outcome: Outcome | float) -> CompositeSignal:
        """Process LLM response and trade outcome.

        Args:
            response_text: The LLM's response text.
            outcome: Trade result as Outcome object or raw PnL float.

        Returns:
            The computed composite reward signal.
        """
        return self._engine.update(response_text, outcome)

    @abstractmethod
    def install(self, target: Any) -> Any:
        """Install this adapter into a framework component.

        Args:
            target: The framework-specific component to wrap
                    (e.g., LangChain chain, CrewAI agent).

        Returns:
            The wrapped component with DopamineCore integrated.
        """
        ...

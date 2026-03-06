"""CrewAI adapter — integrates DopamineCore into CrewAI agents and tasks.

Usage::

    from crewai import Agent, Task, Crew
    from dopamine_core.adapters.crewai import CrewAIAdapter

    adapter = CrewAIAdapter()

    agent = Agent(
        role="BTC Trader",
        goal="Make profitable BTC predictions",
        backstory=adapter.wrap_prompt("You are an expert crypto trader."),
        llm=my_llm,
    )

    # After each task completes, update the engine
    result = crew.kickoff()
    adapter.process_response(result.raw, pnl=0.65)

    # Next task will have updated context
    agent.backstory = adapter.wrap_prompt("You are an expert crypto trader.")

CrewAI is an optional dependency. Install with:
    pip install dopamine-core crewai
"""

from __future__ import annotations

from typing import Any

from dopamine_core.adapters.base import DopamineAdapter
from dopamine_core.config import DopamineConfig


class CrewAIAdapter(DopamineAdapter):
    """Integrates DopamineCore with CrewAI agents.

    CrewAI uses a different paradigm than LangChain — agents have
    persistent backstories and goals rather than per-call prompts.
    This adapter injects DopamineCore context into agent backstories
    and task descriptions.
    """

    def __init__(self, config: DopamineConfig | None = None) -> None:
        super().__init__(config)
        self._original_backstories: dict[int, str] = {}

    def install(self, target: Any) -> Any:
        """Install DopamineCore into a CrewAI Agent.

        Wraps the agent's backstory with reward context. Call this
        before each crew kickoff to update the context based on
        recent performance.

        Args:
            target: A CrewAI Agent instance.

        Returns:
            The same agent with updated backstory.
        """
        agent_id = id(target)

        # Store original backstory on first install
        if agent_id not in self._original_backstories:
            self._original_backstories[agent_id] = getattr(target, "backstory", "")

        original = self._original_backstories[agent_id]
        target.backstory = self.wrap_prompt(original)
        return target

    def wrap_task(self, description: str) -> str:
        """Inject DopamineCore context into a task description.

        Use this when creating new tasks to include current
        reward state in the task framing.

        Args:
            description: Original task description.

        Returns:
            Task description with injected reward context.
        """
        return self.wrap_prompt(description)

    def refresh(self, agent: Any) -> None:
        """Refresh an agent's backstory with current reward state.

        Call this between crew runs to update the agent's context
        based on accumulated performance signals.

        Args:
            agent: A CrewAI Agent instance previously installed.
        """
        self.install(agent)

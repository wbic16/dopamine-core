"""Tests for framework adapters (base, LangChain, CrewAI).

Tests use mock objects to avoid requiring actual framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from dopamine_core.adapters.base import DopamineAdapter
from dopamine_core.adapters.crewai import CrewAIAdapter
from dopamine_core.adapters.langchain import DopamineWrappedLLM, LangChainAdapter
from dopamine_core.types import Outcome


# --- Concrete test adapter ---


class ConcreteAdapter(DopamineAdapter):
    def install(self, target: Any) -> Any:
        return target


class TestDopamineAdapter:
    def test_wrap_prompt_injects_context(self) -> None:
        adapter = ConcreteAdapter()
        result = adapter.wrap_prompt("Analyze BTC")
        assert "[Environmental Context]" in result
        assert "Analyze BTC" in result

    def test_process_response_returns_signal(self) -> None:
        adapter = ConcreteAdapter()
        signal = adapter.process_response("Maybe BTC rises", 0.65)
        assert signal.value is not None

    def test_process_response_with_outcome(self) -> None:
        adapter = ConcreteAdapter()
        outcome = Outcome(pnl=0.65, confidence=0.8)
        signal = adapter.process_response("Confident trade", outcome)
        assert signal.confidence_factor == 0.8

    def test_engine_accessible(self) -> None:
        adapter = ConcreteAdapter()
        assert adapter.engine is not None
        assert adapter.engine.step_count == 0

    def test_state_persists_across_calls(self) -> None:
        adapter = ConcreteAdapter()
        adapter.process_response("trade 1", 0.65)
        adapter.process_response("trade 2", -1.0)
        assert adapter.engine.step_count == 2


# --- LangChain adapter ---


class TestLangChainAdapter:
    def test_install_returns_wrapper(self) -> None:
        adapter = LangChainAdapter()
        mock_llm = MagicMock()
        wrapped = adapter.install(mock_llm)
        assert isinstance(wrapped, DopamineWrappedLLM)

    def test_invoke_injects_context(self) -> None:
        adapter = LangChainAdapter()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "BTC will go UP"
        wrapped = adapter.install(mock_llm)

        result = wrapped.invoke("Predict BTC")

        # Verify the LLM received an augmented prompt
        call_args = mock_llm.invoke.call_args[0][0]
        assert "[Environmental Context]" in call_args
        assert "Predict BTC" in call_args

    def test_invoke_stores_last_response(self) -> None:
        adapter = LangChainAdapter()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "BTC going UP"
        wrapped = adapter.install(mock_llm)

        wrapped.invoke("Predict BTC")
        assert adapter.last_response == "BTC going UP"

    def test_invoke_with_content_attribute(self) -> None:
        adapter = LangChainAdapter()
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "I predict UP"
        mock_llm.invoke.return_value = mock_response
        wrapped = adapter.install(mock_llm)

        wrapped.invoke("Predict BTC")
        assert adapter.last_response == "I predict UP"

    def test_ainvoke_injects_context(self) -> None:
        import asyncio

        adapter = LangChainAdapter()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value="BTC UP")
        wrapped = adapter.install(mock_llm)

        asyncio.run(wrapped.ainvoke("Predict BTC"))

        call_args = mock_llm.ainvoke.call_args[0][0]
        assert "[Environmental Context]" in call_args

    def test_proxy_attributes(self) -> None:
        adapter = LangChainAdapter()
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        wrapped = adapter.install(mock_llm)

        assert wrapped.model_name == "gpt-4"

    def test_underlying_accessible(self) -> None:
        adapter = LangChainAdapter()
        mock_llm = MagicMock()
        wrapped = adapter.install(mock_llm)
        assert wrapped.underlying is mock_llm

    def test_kwargs_passed_through(self) -> None:
        adapter = LangChainAdapter()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "UP"
        wrapped = adapter.install(mock_llm)

        wrapped.invoke("Predict", temperature=0.5, max_tokens=100)
        _, kwargs = mock_llm.invoke.call_args
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 100


# --- CrewAI adapter ---


@dataclass
class MockAgent:
    backstory: str = "You are a trader."
    role: str = "Trader"


class TestCrewAIAdapter:
    def test_install_modifies_backstory(self) -> None:
        adapter = CrewAIAdapter()
        agent = MockAgent()
        adapter.install(agent)
        assert "[Environmental Context]" in agent.backstory
        assert "You are a trader." in agent.backstory

    def test_install_preserves_original(self) -> None:
        adapter = CrewAIAdapter()
        agent = MockAgent(backstory="Original backstory")
        adapter.install(agent)

        # Update engine state
        adapter.process_response("trade", 0.65)

        # Re-install should use original backstory, not the already-modified one
        adapter.install(agent)
        assert agent.backstory.count("[Environmental Context]") == 1

    def test_wrap_task(self) -> None:
        adapter = CrewAIAdapter()
        result = adapter.wrap_task("Analyze the BTC market")
        assert "[Environmental Context]" in result
        assert "Analyze the BTC market" in result

    def test_refresh_updates_context(self) -> None:
        adapter = CrewAIAdapter()
        agent = MockAgent()
        adapter.install(agent)
        initial_backstory = agent.backstory

        # Simulate trades to change state
        for _ in range(5):
            adapter.process_response("Confident prediction", -1.0)

        adapter.refresh(agent)
        # Context should be different after losses
        # (can't guarantee exact text due to random template selection)
        assert "[Environmental Context]" in agent.backstory

    def test_returns_same_agent(self) -> None:
        adapter = CrewAIAdapter()
        agent = MockAgent()
        result = adapter.install(agent)
        assert result is agent

    def test_multiple_agents(self) -> None:
        adapter = CrewAIAdapter()
        agent1 = MockAgent(backstory="Agent 1 backstory")
        agent2 = MockAgent(backstory="Agent 2 backstory")

        adapter.install(agent1)
        adapter.install(agent2)

        assert "Agent 1 backstory" in agent1.backstory
        assert "Agent 2 backstory" in agent2.backstory

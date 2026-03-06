"""LangChain adapter — integrates DopamineCore into LangChain chains and agents.

Usage::

    from langchain_openai import ChatOpenAI
    from dopamine_core.adapters.langchain import LangChainAdapter

    llm = ChatOpenAI(model="gpt-4")
    adapter = LangChainAdapter()
    wrapped_llm = adapter.install(llm)

    # Use wrapped_llm as normal — DopamineCore injects context automatically
    response = wrapped_llm.invoke("Analyze BTC and decide: UP or DOWN?")

    # After getting trade result, update the engine
    adapter.process_response(response.content, pnl=0.65)

LangChain is an optional dependency. Install with:
    pip install dopamine-core langchain-core
"""

from __future__ import annotations

from typing import Any

from dopamine_core.adapters.base import DopamineAdapter
from dopamine_core.config import DopamineConfig


class LangChainAdapter(DopamineAdapter):
    """Wraps a LangChain LLM or ChatModel with DopamineCore context injection.

    The adapter creates a wrapper that intercepts `invoke()` and `ainvoke()`
    calls, injecting subliminal reward context into prompts before they
    reach the LLM.
    """

    def __init__(self, config: DopamineConfig | None = None) -> None:
        super().__init__(config)
        self._last_response: str = ""

    @property
    def last_response(self) -> str:
        return self._last_response

    def install(self, target: Any) -> Any:
        """Wrap a LangChain LLM or ChatModel.

        Args:
            target: A LangChain BaseLLM or BaseChatModel instance.

        Returns:
            A DopamineWrappedLLM that proxies all calls through DopamineCore.
        """
        return DopamineWrappedLLM(target, self)


class DopamineWrappedLLM:
    """Proxy that injects DopamineCore context into LangChain LLM calls.

    Wraps the underlying LLM's invoke/ainvoke methods to inject
    reward-based context before each call.
    """

    def __init__(self, llm: Any, adapter: LangChainAdapter) -> None:
        self._llm = llm
        self._adapter = adapter

    @property
    def underlying(self) -> Any:
        return self._llm

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        """Synchronous invoke with context injection.

        Args:
            input: String prompt or LangChain message(s).
            **kwargs: Passed through to the underlying LLM.

        Returns:
            LLM response (same type as underlying LLM returns).
        """
        modified_input = self._inject_into_input(input)
        response = self._llm.invoke(modified_input, **kwargs)
        self._adapter._last_response = self._extract_text(response)
        return response

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        """Async invoke with context injection.

        Args:
            input: String prompt or LangChain message(s).
            **kwargs: Passed through to the underlying LLM.

        Returns:
            LLM response (same type as underlying LLM returns).
        """
        modified_input = self._inject_into_input(input)
        response = await self._llm.ainvoke(modified_input, **kwargs)
        self._adapter._last_response = self._extract_text(response)
        return response

    def _inject_into_input(self, input: Any) -> Any:
        """Inject DopamineCore context into the input.

        Handles both string inputs and LangChain message lists.
        """
        if isinstance(input, str):
            return self._adapter.wrap_prompt(input)

        # Handle LangChain message lists
        if isinstance(input, list) and len(input) > 0:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
            except ImportError:
                return input

            # Find the last human message and inject context
            for i in range(len(input) - 1, -1, -1):
                if isinstance(input[i], HumanMessage):
                    modified = list(input)
                    original_content = input[i].content
                    if isinstance(original_content, str):
                        modified[i] = HumanMessage(
                            content=self._adapter.wrap_prompt(original_content)
                        )
                    return modified

            # No human message found — prepend as system message
            context = self._adapter.engine.inject_context("")
            modified = list(input)
            modified.insert(0, SystemMessage(content=context))
            return modified

        return input

    def _extract_text(self, response: Any) -> str:
        """Extract text content from various LangChain response types."""
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            content = response.content
            return content if isinstance(content, str) else str(content)
        return str(response)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying LLM."""
        return getattr(self._llm, name)

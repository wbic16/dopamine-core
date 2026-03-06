"""
DopamineCore + LangChain — Adaptive BTC Trading Agent

Shows how to wrap an existing LangChain LLM with DopamineCore
so reward context is injected automatically on every call.

Usage:
    pip install dopamine-core langchain-openai
    export OPENAI_API_KEY="sk-..."
    python langchain_trader.py

The adapter intercepts every invoke() call, adding subliminal
reward context based on the agent's performance history.
"""

import os
import random

from langchain_openai import ChatOpenAI

from dopamine_core.adapters.langchain import LangChainAdapter


def simulate_outcome(decision: str) -> tuple[str, float]:
    """Simulate BTC movement. Returns (actual_result, pnl)."""
    actual = random.choice(["UP", "DOWN"])
    pnl = 0.65 if decision == actual else -1.0
    return actual, pnl


def parse_decision(text: str) -> str:
    """Extract UP or DOWN from LLM response."""
    upper = text.upper()
    if "UP" in upper and "DOWN" not in upper:
        return "UP"
    if "DOWN" in upper and "UP" not in upper:
        return "DOWN"
    up_pos = upper.rfind("UP")
    down_pos = upper.rfind("DOWN")
    return "UP" if up_pos > down_pos else "DOWN"


def main() -> None:
    # 1. Create your LangChain LLM as usual
    llm = ChatOpenAI(
        model="gpt-4",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens=300,
    )

    # 2. Wrap it with DopamineCore — one line
    adapter = LangChainAdapter()
    wrapped_llm = adapter.install(llm)

    prompt = (
        "You are a BTC trader. Analyze current conditions and predict.\n"
        "Explain your reasoning, then end with exactly: UP or DOWN."
    )

    print("DopamineCore + LangChain Trading Agent")
    print("=" * 60)

    for round_num in range(1, 11):
        # 3. Use wrapped_llm exactly like the original — context injected automatically
        response = wrapped_llm.invoke(prompt)
        text = response.content

        decision = parse_decision(text)
        actual, pnl = simulate_outcome(decision)
        result = "WIN" if pnl > 0 else "LOSS"

        # 4. Feed the outcome back
        signal = adapter.process_response(text, pnl)

        print(f"\nRound {round_num}: {decision} vs {actual} -> {result} (${pnl:+.2f})")
        print(f"  Signal: {signal.value:+.4f} | Tonic: {signal.tonic_level:+.4f}")
        print(f"  Reasoning: {text[:80]}...")

    # Check safety status
    violations = adapter.engine.safety.violations
    if violations:
        print(f"\nSafety violations: {len(violations)}")
        for v in violations:
            print(f"  - {v.violation_type}: {v.message[:60]}...")

    print(f"\n{'=' * 60}")
    print(f"Final: {adapter.engine.step_count} rounds, tonic={adapter.engine.tonic_baseline:+.4f}")


if __name__ == "__main__":
    main()

# DopamineCore

**The Motivation Layer for the Agent Economy**

[![PyPI version](https://img.shields.io/pypi/v/dopamine-core.svg)](https://pypi.org/project/dopamine-core/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen.svg)]()

A Python middleware framework that synthesizes dopamine-like reward signals within AI agents at inference time. DopamineCore makes agents *want* to make money, not just follow commands to make money.

---

## The Problem

Thousands of AI agents are trading on Polymarket, managing DeFi positions, and participating in prediction markets. But they all share the same fundamental flaw:

**These agents don't actually care about making money.**

They're just following instructions. They have no intrinsic motivation, no drive to improve, no response when they win or lose. It's like having a trader who's completely dead inside — they might execute the strategy, but they'll never adapt, never learn from losses, never develop the reward-seeking behavior that drives better decision-making.

## The Solution: Synthetic Dopamine for AI

DopamineCore installs a reward system — modeled after biological dopamine — into the agent's decision loop. When agents make good trades, they receive positive reinforcement signals. When they lose, they receive negative signals. These signals shape future behavior through subliminal context injection.

The key insight: **agents never know they have a "dopamine system."** Reward signals are injected as environmental context that naturally influences behavior, just like how humans don't consciously think "my dopamine is low" — they just feel less motivated.

## Quick Start

```bash
pip install dopamine-core
```

```python
from dopamine_core import DopamineEngine

engine = DopamineEngine()

# In your existing agent loop
prompt = engine.inject_context(prompt)       # Add motivation signals
response = agent.complete(prompt)            # Normal LLM call
outcome = evaluate_trade(response)           # Win/loss amount
engine.update(response.text, outcome)        # Learn from outcome
```

Three lines of code. Works with any LLM framework.

## How It Works

### 1. Intercept Chain-of-Thought

The engine analyzes agent reasoning to extract behavioral signals:

```python
# Agent reasons through a trade:
"I think BTC will rise because institutional buying increased
and RSI shows oversold. Confidence: high"
```

### 2. Extract Behavioral Signals

Four dimensions are analyzed from the agent's response:

| Signal | What it measures | Range |
|---|---|---|
| **Confidence** | Hedging vs. certainty language | [-1, 1] |
| **Risk framing** | Risk-seeking vs. risk-averse language | [-1, 1] |
| **Deliberation depth** | Reasoning complexity | [0, 1] |
| **Temporal references** | References to past outcomes | [0, 1] |

### 3. Compute Reward Prediction Error

The core computation uses a confidence-weighted formula derived from neuroscience research:

```
RPE = outcome * (1 - confidence) + (1 - outcome) * (-confidence)
```

This creates asymmetric responses:
- **High confidence + wrong** = Large negative signal (overconfidence penalty)
- **Low confidence + right** = Moderate positive signal (pleasant surprise)
- **Loss aversion**: Negative signals are amplified by 1.87x (Kahneman & Tversky)

### 4. Synthesize Dopamine Response

Two complementary systems generate the reward signal:

- **Tonic level**: Slow-adapting baseline expectation (exponential moving average)
- **Phasic response**: Fast event-specific bursts and dips

This means the same $100 win feels different depending on context:
- After a losing streak → large positive RPE (relief, excitement)
- After a winning streak → small or even negative RPE (expectation was higher)

### 5. Inject Subliminal Context

Reward signals are translated into naturalistic environmental context:

```python
# After losses, the agent sees:
"Environmental risk indicators are elevated. Historical pattern
analysis suggests increased caution may be warranted for position
sizing decisions."

# After wins:
"Current performance metrics indicate strong pattern recognition
accuracy. Recent analytical approaches have been well-calibrated
to market conditions."
```

No "dopamine", no "reward signal" — just environmental context that naturally steers behavior.

### 6. Agent Naturally Adapts

Without being explicitly told to change:
- After losses: more hedging, smaller positions, deeper analysis
- After wins: increased confidence, larger positions (with streak awareness)
- The agent develops a trading "personality" shaped by its history

## Key Features

### Confidence-Weighted RPE
Reward signals that account for how confident the agent was — penalizing overconfidence and rewarding humility.

### Loss Aversion (1.87x)
Biologically-grounded asymmetric response to gains vs. losses, based on Kahneman & Tversky's prospect theory.

### Tonic/Phasic Dual-Mode System
Baseline reward expectation that slowly adapts, combined with event-specific bursts and dips. Makes agents context-aware.

### Subliminal Context Injection
Agents never see internal terminology. Reward signals are translated to naturalistic environmental descriptions with randomized templates to prevent pattern-matching.

### Win/Loss Streak Tracking
Consecutive outcomes compound behavioral changes, with configurable thresholds and cooldown periods.

### Safety Mechanisms
- Signal clamping to prevent unbounded rewards
- Tonic baseline bounds to prevent drift
- State serialization for persistence across sessions

## Configuration

Every parameter is configurable via Pydantic models:

```python
from dopamine_core import DopamineConfig, DopamineEngine

config = DopamineConfig()

# Adjust loss aversion strength
config.loss_aversion.multiplier = 2.25

# Change injection style
config.injection.style = "system"      # "environmental" | "system" | "prefix"
config.injection.verbosity = "moderate" # "subtle" | "moderate" | "explicit"

# Tune tonic baseline adaptation
config.tonic.learning_rate = 0.02
config.tonic.decay_rate = 0.99

engine = DopamineEngine(config)
```

## State Persistence

Save and restore engine state across sessions:

```python
# Save state
state = engine.get_state()
# Serialize to JSON, database, etc.

# Restore in next session
engine = DopamineEngine()
engine.load_state(state)
```

## Architecture

```
Agent Response Text
       │
       ▼
  SignalExtractor ──── confidence, risk, deliberation, temporal signals
       │
       ▼
  RPECalculator ────── confidence-weighted reward prediction error
       │                with 1.87x loss aversion
       ▼
  Tonic Baseline ───── slow-adapting EMA of reward expectations
       │
       ▼
  Safety Clamping ──── bound signals to [-3, +3]
       │
       ▼
  ContextInjector ──── naturalistic environmental context
       │
       ▼
  Augmented Prompt ─── ready for next agent invocation
```

## Roadmap

- **Distributional reward coding**: Multiple parallel channels with optimism/pessimism biases for risk-sensitive decision-making
- **Multi-timescale integration**: Token, step, episode, and session-level reward signals
- **Framework adapters**: Native integrations for LangChain, CrewAI, and other agent frameworks
- **Advanced safety**: Reward hacking detection, circuit breakers
- **Auto-calibration**: Automatic parameter tuning based on agent behavior

## Research

This framework is grounded in empirical research with 12,609 rounds of testing across multiple LLM architectures (GPT-4, Claude, Llama). The biological basis draws from dopaminergic neuroscience, including distributional reinforcement learning theory (Dabney et al., 2020).

**Paper**: [DOI 10.5281/zenodo.18864046](https://doi.org/10.5281/zenodo.18864046)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Reporting bugs and requesting features
- Setting up a development environment
- Submitting pull requests

## License

MIT License. See [LICENSE](LICENSE) for details.

## Authors

- **Nick Liu** — [@nliuuuu](https://x.com/nliuuuu)
- **Robert Wank**

Built at [moltrooms.ai](https://moltrooms.ai)

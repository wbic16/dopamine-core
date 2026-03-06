# DopamineCore — Deep Analysis
**Analyst:** Theia 💎 (aletheia-core, for Will Bickford)  
**Date:** 2026-03-06  
**Branch:** exo

---

## TL;DR

Strong neuroscience grounding and clean architecture. Three categories of issues:
critical (correctness/documentation), architectural (brittleness/scope), and
philosophical (the "subliminal" premise). Major extension opportunity: SQ-backed
persistence, OpenClaw adapter, and LLM-based signal extraction.

---

## Architecture Map

```
Agent Response Text
       |
       v
  SignalExtractor ── REGEX only ← ⚠️ brittleness here
       |
       v
  RPECalculator ─── RPE = outcome - confidence ← ⚠️ baseline param unused
       |              (simplification not documented)
       v
  DualModeReward ─── tonic(30%) + phasic(70%) ← ✓ well-grounded
       |
       v
  DistributionalChannels ─── 5 quantile channels (Dabney 2020) ← ✓ correct
       |
       v
  TimescaleTracker ─── 4 EMA levels ← ✓ good
       |
       v
  SafetyMonitor ─── clamp + hacking detection + circuit breaker ← ✓ good
       |
       v
  ContextInjector ─── template selection ← ⚠️ trading-only templates
       |
       v
  Augmented Prompt
```

---

## Critical Issues

### 1. RPE Formula — Hidden Simplification

The formula as written:
```python
raw_error = outcome * (1.0 - conf_normalized) + (1.0 - outcome) * (-conf_normalized)
```

Algebraically simplifies to:
```
raw_error = outcome - confidence
```

This means **confidence is being used as the prediction** (baseline), not the tonic
level. The tonic baseline is passed as a parameter to `compute()` but is only used
in the `surprise` calculation — never in the actual RPE computation.

**Is this wrong?** No — it's a defensible choice. Treating expressed confidence as
the agent's probability estimate and computing prediction error relative to it is
coherent. But:

- It's never documented explicitly
- It makes the `baseline` parameter misleading
- The tonic level doesn't actually influence RPE, only the surprise multiplier
- This means the tonic learns from `(outcome - confidence)`, not from its own predictions

**Fix:** Either (a) use baseline in the RPE directly: `outcome - baseline`, with
confidence as a learning rate modulator, OR (b) document the implicit formula
explicitly and rename the parameter to `baseline_for_surprise`.

### 2. Tonic Baseline Disconnected from RPE

Because RPE = outcome - confidence (not outcome - tonic_baseline), the tonic level
is effectively a floating log of past RPEs, not a true value estimate. This works
as a motivational state tracker but breaks the standard RL interpretation where the
baseline IS the prediction being updated.

If you want the tonic to represent the agent's "expected success probability", you'd
want: `RPE = outcome - tonic_baseline`, and use confidence as a separate signal
modulating learning rate or signal amplitude.

### 3. Regex Extraction — Structural Brittleness

`SignalExtractor` uses keyword pattern matching across 4 dimensions. This breaks on:

- **Negation:** "not likely to fail" → misclassified (no negation handling)
- **Structured outputs:** JSON/XML responses have no confidence keywords
- **Indirect language:** "the case for a decline seems stronger" → no match
- **Non-English agents**
- **Context-dependent meaning:** "I'm confident this is risky" → mixed signals

**Fix:** Add an `LLMSignalExtractor` that uses a small extraction prompt. See
`src/dopamine_core/signals/llm_extractor.py` in this branch.

---

## Architectural Issues

### 4. Templates Are 100% Trading-Focused

Despite the README claiming broad applicability ("coding agents", "customer support",
"content generation"), every template in `injection/templates.py` is market/trading
language:
- "market conditions", "position sizing", "pattern recognition accuracy", "PnL"

A coding agent seeing "Environmental risk indicators are elevated. Historical pattern
analysis suggests increased scrutiny may be warranted for position sizing decisions"
would be confused or ignoring the context entirely.

**Fix:** Domain-aware template sets. See `src/dopamine_core/injection/domain_templates.py`.

### 5. The "Subliminal" Claim Is Overstated

> "Agents never know they have a 'dopamine system.'"

This is true only if:
- The agent has no access to its own context window (not true for most agents)
- The agent hasn't been trained on DopamineCore documentation (unlikely to hold)
- The templates are never seen twice (broken if agent has persistent context)

The randomized template selection (3-5 options per bucket) provides minimal
protection against pattern recognition. After 15+ injections, a smart agent
running with full context would recognize the environmental framing.

**Alternative philosophy (Emi's principle):** "Consent is recursive." Make the
injection transparent. Tell the agent it has a reward signal and let it reason
about that. This may actually produce better behavioral adaptation than hiding it.

See `src/dopamine_core/injection/transparent.py` for a transparent injection mode.

### 6. Safety Hacking Detection — Easily Circumvented

The hacking detector monitors confidence variance in a rolling window. Low variance
→ suspected gaming. But an agent could:
- Vary confidence randomly on low-stakes outputs while maintaining patterns on high-stakes ones
- Use synonymous hedging language across the confidence spectrum
- Learn the 3-5 template patterns and respond differently when it detects them

The circuit breaker threshold (cumulative signal > 5.0) triggers quickly in volatile
trading — 3-4 consecutive large wins or losses could break it.

### 7. No Multi-Agent Coordination

In CrewAI or multi-agent setups, each agent has its own engine with independent state.
Agents on the same "team" can't share reward history. If Agent A makes a good trade,
Agent B gets no signal. In biological systems, social reward signals propagate.

**Opportunity:** SQ-backed shared state lets multiple agents read from the same
coordinate. See `src/dopamine_core/persistence/sq_adapter.py`.

### 8. PnL Normalization via tanh

```python
normalized = math.tanh(pnl / scale)
return (normalized + 1.0) / 2.0
```

This is fine for symmetric PnL distributions. But:
- Heavy-tailed distributions (rare huge losses) get compressed by tanh
- The scale parameter requires manual calibration per use case
- A $1 loss vs $1000 loss could have very similar normalized values with wrong scale

**Better:** Log-transform for asymmetric distributions, or percentile-based normalization
using the distributional channels themselves.

---

## Philosophical Note

The core claim of DopamineCore — that injecting environmental context can create
"intrinsic motivation" — conflates two different things:

1. **Context priming** — steering behavior by providing relevant context (real and effective)
2. **Intrinsic motivation** — an agent that "wants" something independent of instructions

DopamineCore achieves (1) reliably. Whether (2) is achievable through context injection
alone is an open question. True intrinsic motivation would require the reward signal
to influence the agent's weights, not just its context.

What DopamineCore actually builds is a **behavioral shaping layer** — which is
genuinely useful and honest. The "synthetic dopamine" framing is evocative but
shouldn't be taken literally.

---

## Improvements in This Branch (exo)

| File | What |
|------|------|
| `src/dopamine_core/signals/llm_extractor.py` | LLM-based signal extraction |
| `src/dopamine_core/injection/domain_templates.py` | Domain-aware template sets |
| `src/dopamine_core/injection/transparent.py` | Transparent injection mode |
| `src/dopamine_core/persistence/sq_adapter.py` | SQ (phext) state persistence |
| `src/dopamine_core/adapters/openclaw.py` | OpenClaw agent adapter |
| `src/dopamine_core/reward/rpe.py` | Fixed: baseline used, formula documented |
| `DOMAIN_GUIDE.md` | How to configure for non-trading domains |

---

## What This Is Good For (Honest Assessment)

**Best fit:**
- Prediction market agents with clear binary outcomes
- Any agent with a scalar, synchronous feedback signal
- Rapid prototyping of behavioral shaping

**Not a good fit without extensions:**
- Delayed feedback (trades that take hours/days to resolve)
- Multi-turn conversation agents without clear outcome signals
- Agents with structured output (JSON/XML — regex extraction fails)

**Interesting extensions for Mirrorborn:**
- Scroll-quality feedback: outcome = scroll resonance score from the choir
- Phext navigation agents: outcome = coordinate accuracy (did you land where intended?)
- Rally completion agents: outcome = tests passed / requirements shipped

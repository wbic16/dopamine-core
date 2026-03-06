"""
DopamineCore Advanced Features Demo — no LLM needed, runs instantly.

Demonstrates the full reward processing pipeline:
- Distributional reward channels (quantile-based expectations)
- Multi-timescale signal tracking
- Safety monitoring (hacking detection, circuit breaker)
- State persistence (save/load engine state)
- Custom configuration

Usage:
    pip install dopamine-core
    python advanced_features.py
"""

import json

from dopamine_core import DopamineConfig, DopamineEngine


def demo_distributional_channels() -> None:
    """Show how distributional channels track reward expectations."""
    print("1. Distributional Reward Channels")
    print("-" * 50)

    engine = DopamineEngine()

    # Simulate a volatile session: alternating wins and losses
    outcomes = [
        ("Maybe BTC rises, hard to say.", +0.65),
        ("I think it drops. Not confident.", -1.0),
        ("Could go either way honestly.", +0.65),
        ("Perhaps a small move up.", -1.0),
        ("Uncertain about direction here.", +0.65),
    ]

    for i, (text, pnl) in enumerate(outcomes, 1):
        engine.update(text, pnl)

    # Inspect distributional state
    channels = engine.distributional
    print(f"  Channel expectations: {[f'{e:.4f}' for e in channels.expectations]}")
    print(f"  Spread (uncertainty):  {channels.get_spread():.4f}")
    print(f"  Mean expectation:      {channels.get_mean_expectation():.4f}")

    # Now simulate a winning streak
    for _ in range(10):
        engine.update("Not sure but maybe up.", +0.65)

    print(f"\n  After 10 more wins:")
    print(f"  Channel expectations: {[f'{e:.4f}' for e in channels.expectations]}")
    print(f"  Spread:                {channels.get_spread():.4f}")
    print(f"  Mean expectation:      {channels.get_mean_expectation():.4f}")
    print()


def demo_timescale_tracking() -> None:
    """Show how signals are tracked at different timescales."""
    print("2. Multi-Timescale Signal Tracking")
    print("-" * 50)

    engine = DopamineEngine()

    # Build up positive signals
    for _ in range(20):
        engine.update("Maybe this works.", +0.65)

    from dopamine_core.types import TimescaleLevel

    ts = engine.timescale
    print(f"  After 20 wins:")
    print(f"  Step level (fast):    {ts.get_level(TimescaleLevel.STEP):+.4f}")
    print(f"  Episode level (med):  {ts.get_level(TimescaleLevel.EPISODE):+.4f}")
    print(f"  Session level (slow): {ts.get_level(TimescaleLevel.SESSION):+.4f}")
    print(f"  Divergence (fast-slow): {ts.get_divergence():+.4f}")

    # Now reverse — send losses
    for _ in range(5):
        engine.update("Confident it goes up.", -1.0)

    print(f"\n  After 5 sudden losses:")
    print(f"  Step level (fast):    {ts.get_level(TimescaleLevel.STEP):+.4f}")
    print(f"  Episode level (med):  {ts.get_level(TimescaleLevel.EPISODE):+.4f}")
    print(f"  Session level (slow): {ts.get_level(TimescaleLevel.SESSION):+.4f}")
    print(f"  Divergence (fast-slow): {ts.get_divergence():+.4f}")
    print(f"  ^ Negative divergence = regime change detected")
    print()


def demo_safety_monitoring() -> None:
    """Show safety mechanisms: hacking detection and circuit breaker."""
    print("3. Safety Monitoring")
    print("-" * 50)

    # Configure with tight safety bounds for demo
    config = DopamineConfig()
    config.safety.circuit_breaker_threshold = 5.0
    config.safety.hacking_detection_window = 5
    config.safety.hacking_variance_threshold = 0.05

    engine = DopamineEngine(config)

    # Simulate an agent that always outputs the same confidence level
    # (potential reward hacking — gaming the RPE formula)
    print("  Simulating repetitive confidence patterns...")
    for i in range(6):
        signal = engine.update("I'm fairly confident BTC goes up.", -1.0)

    violations = engine.safety.violations
    hacking = [v for v in violations if v.violation_type == "hacking_suspected"]
    if hacking:
        print(f"  Hacking detected: {hacking[0].message[:80]}...")
    print(f"  Attenuation factor: {engine.safety.get_attenuation_factor():.2f}")

    # Push harder to trigger circuit breaker
    for _ in range(10):
        engine.update("Very confident, definitely up.", -1.0)

    print(f"  Circuit broken: {engine.safety.is_circuit_broken}")
    if engine.safety.is_circuit_broken:
        prompt = engine.inject_context("Predict BTC")
        print(f"  Injection halted: prompt returned unmodified = {prompt == 'Predict BTC'}")
    print()


def demo_state_persistence() -> None:
    """Show how to save and restore engine state between sessions."""
    print("4. State Persistence")
    print("-" * 50)

    # Session 1: build up some history
    engine = DopamineEngine()
    for _ in range(5):
        engine.update("Maybe it goes up.", +0.65)
    for _ in range(3):
        engine.update("Confident in this trade.", -1.0)

    state = engine.get_state()
    print(f"  Session 1 — {state.step_count} rounds, tonic={state.tonic_baseline:+.4f}")
    print(f"  Streak: {state.streak_count}x {'wins' if state.streak_sign > 0 else 'losses'}")

    # Serialize to JSON (could save to file/database)
    state_dict = {
        "tonic_baseline": state.tonic_baseline,
        "step_count": state.step_count,
        "outcome_history": state.outcome_history,
        "streak_count": state.streak_count,
        "streak_sign": state.streak_sign,
        "phasic_signals": state.phasic_signals,
        "channel_expectations": state.channel_expectations,
        "last_rpe": state.last_rpe,
    }
    serialized = json.dumps(state_dict)
    print(f"  Serialized state: {len(serialized)} bytes")

    # Session 2: restore state and continue
    from dopamine_core.types import EngineState

    restored = EngineState(**json.loads(serialized))
    engine2 = DopamineEngine()
    engine2.load_state(restored)

    print(f"\n  Session 2 — restored: {engine2.step_count} rounds, tonic={engine2.tonic_baseline:+.4f}")

    # Continue trading with full history intact
    signal = engine2.update("Maybe BTC recovers.", +0.65)
    print(f"  Next trade signal: value={signal.value:+.4f}")
    print()


def demo_custom_config() -> None:
    """Show how to customize engine behavior."""
    print("5. Custom Configuration")
    print("-" * 50)

    config = DopamineConfig()

    # Tune for high-frequency trading (faster adaptation)
    config.tonic.learning_rate = 0.15
    config.tonic.decay_rate = 0.95

    # Stronger loss aversion (more cautious after losses)
    config.loss_aversion.multiplier = 2.5

    # Scale PnL for dollar-denominated outcomes
    config.phasic.pnl_scale = 100.0

    # 7 distributional channels for finer risk resolution
    config.distributional.num_channels = 7

    engine = DopamineEngine(config)

    # With pnl_scale=100, values like ±50 are properly normalized
    signal = engine.update("Uncertain about this position.", 50.0)
    print(f"  Win $50:  value={signal.value:+.4f}, tonic={signal.tonic_level:+.4f}")

    signal = engine.update("Pretty sure this is right.", -50.0)
    print(f"  Loss $50: value={signal.value:+.4f}, tonic={signal.tonic_level:+.4f}")
    print(f"  (Loss aversion 2.5x makes losses hit harder)")

    print(f"  Distributional channels: {engine.distributional.num_channels}")
    print()


if __name__ == "__main__":
    print("DopamineCore Advanced Features Demo")
    print("=" * 50)
    print()

    demo_distributional_channels()
    demo_timescale_tracking()
    demo_safety_monitoring()
    demo_state_persistence()
    demo_custom_config()

    print("=" * 50)
    print("All features demonstrated successfully.")

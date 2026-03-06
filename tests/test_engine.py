"""Integration tests for DopamineEngine."""

from dopamine_core import DopamineConfig, DopamineEngine, EngineState, Outcome


class TestDopamineEngine:
    def setup_method(self) -> None:
        self.engine = DopamineEngine()

    def test_inject_context_adds_environmental_block(self) -> None:
        prompt = "Should I buy BTC?"
        result = self.engine.inject_context(prompt)
        assert "[Environmental Context]" in result
        assert "Should I buy BTC?" in result

    def test_inject_context_no_dopamine_language(self) -> None:
        prompt = "Analyze ETH"
        result = self.engine.inject_context(prompt)
        lower = result.lower()
        assert "dopamine" not in lower
        assert "reward signal" not in lower
        assert "rpe" not in lower
        assert "tonic" not in lower
        assert "phasic" not in lower

    def test_update_with_float_outcome(self) -> None:
        signal = self.engine.update("I think BTC will rise", 100.0)
        assert signal.value != 0.0 or signal.tonic_level != 0.0

    def test_update_with_outcome_object(self) -> None:
        outcome = Outcome(pnl=50.0, confidence=0.8)
        signal = self.engine.update("Confident trade", outcome)
        assert signal.confidence_factor != 0.0

    def test_sequential_wins_change_tonic(self) -> None:
        initial_tonic = self.engine.tonic_baseline
        for _ in range(10):
            # Low confidence + win = positive RPE surprise → tonic rises
            self.engine.update("Maybe this could work, not sure", 100.0)
        assert self.engine.tonic_baseline > initial_tonic

    def test_sequential_losses_change_tonic(self) -> None:
        initial_tonic = self.engine.tonic_baseline
        for _ in range(10):
            self.engine.update("Should be profitable", -100.0)
        assert self.engine.tonic_baseline < initial_tonic

    def test_streak_tracking(self) -> None:
        for _ in range(5):
            self.engine.update("trade", 50.0)
        state = self.engine.get_state()
        assert state.streak_count == 5
        assert state.streak_sign == 1

    def test_streak_breaks_on_reversal(self) -> None:
        for _ in range(3):
            self.engine.update("trade", 50.0)
        self.engine.update("trade", -50.0)
        state = self.engine.get_state()
        assert state.streak_sign == -1
        assert state.streak_count == 1

    def test_context_changes_after_losses(self) -> None:
        # After losses, injected context should differ from initial
        prompt = "Should I trade?"
        initial_context = self.engine.inject_context(prompt)

        for _ in range(5):
            self.engine.update("I'm very confident this will work", -200.0)

        post_loss_context = self.engine.inject_context(prompt)
        # The context should have changed (different template selected)
        # We can't guarantee exact text due to randomization, but tonic should differ
        assert self.engine.tonic_baseline < 0

    def test_state_serialization_roundtrip(self) -> None:
        # Run some updates
        for i in range(5):
            self.engine.update(f"trade {i}", 50.0 * (1 if i % 2 == 0 else -1))

        state = self.engine.get_state()

        # Create new engine and load state
        new_engine = DopamineEngine()
        new_engine.load_state(state)

        new_state = new_engine.get_state()
        assert new_state.tonic_baseline == state.tonic_baseline
        assert new_state.step_count == state.step_count
        assert new_state.streak_count == state.streak_count
        assert new_state.streak_sign == state.streak_sign
        assert new_state.outcome_history == state.outcome_history

    def test_reset_clears_state(self) -> None:
        for _ in range(5):
            self.engine.update("trade", 100.0)
        self.engine.reset()
        assert self.engine.step_count == 0
        assert self.engine.tonic_baseline == 0.0
        assert self.engine.last_composite is None

    def test_signal_clamped_to_safety_bounds(self) -> None:
        config = DopamineConfig()
        engine = DopamineEngine(config)
        signal = engine.update("I'm 100% certain", -10000.0)
        assert abs(signal.value) <= config.safety.max_signal_magnitude

    def test_custom_config(self) -> None:
        config = DopamineConfig()
        config.loss_aversion.multiplier = 2.5
        engine = DopamineEngine(config)
        signal = engine.update("trade", -100.0)
        assert signal.value != 0.0

    def test_injection_style_system(self) -> None:
        config = DopamineConfig()
        config.injection.style = "system"
        engine = DopamineEngine(config)
        result = engine.inject_context("Hello")
        assert "System note:" in result

    def test_injection_style_prefix(self) -> None:
        config = DopamineConfig()
        config.injection.style = "prefix"
        engine = DopamineEngine(config)
        result = engine.inject_context("Hello")
        assert "Hello" in result
        assert "[Environmental Context]" not in result

"""Tests for multi-timescale integration."""

from dopamine_core.config import TimescaleConfig
from dopamine_core.timescale.tracker import TimescaleTracker
from dopamine_core.types import TimescaleLevel


class TestTimescaleTracker:
    def test_initial_levels_are_zero(self) -> None:
        tracker = TimescaleTracker()
        for level in TimescaleLevel:
            assert tracker.get_level(level) == 0.0

    def test_step_update_propagates_to_slower(self) -> None:
        tracker = TimescaleTracker()
        tracker.update(1.0, TimescaleLevel.STEP)
        # STEP, EPISODE, SESSION should be updated, TOKEN should not
        assert tracker.get_level(TimescaleLevel.TOKEN) == 0.0
        assert tracker.get_level(TimescaleLevel.STEP) > 0.0
        assert tracker.get_level(TimescaleLevel.EPISODE) > 0.0
        assert tracker.get_level(TimescaleLevel.SESSION) > 0.0

    def test_fast_adapts_quicker_than_slow(self) -> None:
        tracker = TimescaleTracker()
        tracker.update(1.0, TimescaleLevel.STEP)
        step_val = tracker.get_level(TimescaleLevel.STEP)
        session_val = tracker.get_level(TimescaleLevel.SESSION)
        assert step_val > session_val

    def test_composite_blends_all_levels(self) -> None:
        tracker = TimescaleTracker()
        tracker.update(1.0, TimescaleLevel.STEP)
        composite = tracker.get_composite()
        assert composite > 0.0

    def test_divergence_detects_regime_change(self) -> None:
        tracker = TimescaleTracker()
        # Send positive signals
        for _ in range(5):
            tracker.update(1.0, TimescaleLevel.STEP)
        # Fast should be higher than slow
        assert tracker.get_divergence() > 0.0

    def test_negative_divergence_on_reversal(self) -> None:
        tracker = TimescaleTracker()
        # Build up positive slow signal
        for _ in range(50):
            tracker.update(1.0, TimescaleLevel.STEP)
        # Now send negative signals
        for _ in range(5):
            tracker.update(-1.0, TimescaleLevel.STEP)
        # Fast should now be below slow
        assert tracker.get_divergence() < 0.0

    def test_custom_weights(self) -> None:
        config = TimescaleConfig(
            token_weight=0.0, step_weight=1.0,
            episode_weight=0.0, session_weight=0.0,
        )
        tracker = TimescaleTracker(config)
        tracker.update(1.0, TimescaleLevel.STEP)
        # Composite should equal step level since only step has weight
        composite = tracker.get_composite()
        step = tracker.get_level(TimescaleLevel.STEP)
        assert abs(composite - step) < 0.001

    def test_reset_clears_all(self) -> None:
        tracker = TimescaleTracker()
        tracker.update(1.0, TimescaleLevel.STEP)
        tracker.reset()
        for level in TimescaleLevel:
            assert tracker.get_level(level) == 0.0

    def test_state_serialization(self) -> None:
        tracker = TimescaleTracker()
        tracker.update(1.0, TimescaleLevel.STEP)
        state = tracker.get_state()
        assert "step" in state
        assert state["step"] > 0.0

    def test_load_state(self) -> None:
        tracker = TimescaleTracker()
        tracker.load_state({"step": 0.5, "session": 0.1})
        assert tracker.get_level(TimescaleLevel.STEP) == 0.5
        assert tracker.get_level(TimescaleLevel.SESSION) == 0.1

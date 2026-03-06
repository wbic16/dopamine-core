"""Tests for momentum tracking."""

from dopamine_core.config import MomentumConfig
from dopamine_core.signals.momentum import MomentumTracker


class TestMomentumTracker:
    def test_initial_state(self) -> None:
        tracker = MomentumTracker()
        assert tracker.streak_count == 0
        assert tracker.streak_sign == 0
        assert tracker.get_momentum_factor() == 0.0

    def test_win_streak_builds(self) -> None:
        tracker = MomentumTracker()
        for _ in range(5):
            tracker.update(1.0)
        assert tracker.streak_count == 5
        assert tracker.streak_sign == 1

    def test_loss_streak_builds(self) -> None:
        tracker = MomentumTracker()
        for _ in range(4):
            tracker.update(-1.0)
        assert tracker.streak_count == 4
        assert tracker.streak_sign == -1

    def test_streak_breaks_on_reversal(self) -> None:
        tracker = MomentumTracker()
        for _ in range(3):
            tracker.update(1.0)
        tracker.update(-1.0)
        assert tracker.streak_count == 1
        assert tracker.streak_sign == -1

    def test_zero_pnl_does_not_affect_streak(self) -> None:
        tracker = MomentumTracker()
        tracker.update(1.0)
        tracker.update(0.0)
        assert tracker.streak_count == 1

    def test_momentum_factor_below_threshold_is_zero(self) -> None:
        tracker = MomentumTracker()
        tracker.update(1.0)
        tracker.update(1.0)
        assert tracker.get_momentum_factor() == 0.0

    def test_momentum_factor_at_threshold(self) -> None:
        tracker = MomentumTracker(MomentumConfig(streak_threshold=3))
        for _ in range(3):
            tracker.update(1.0)
        factor = tracker.get_momentum_factor()
        assert factor > 0

    def test_negative_momentum_for_loss_streak(self) -> None:
        tracker = MomentumTracker(MomentumConfig(streak_threshold=3))
        for _ in range(3):
            tracker.update(-1.0)
        factor = tracker.get_momentum_factor()
        assert factor < 0

    def test_momentum_capped_at_max(self) -> None:
        config = MomentumConfig(streak_threshold=2, max_streak_multiplier=1.5)
        tracker = MomentumTracker(config)
        for _ in range(100):
            tracker.update(1.0)
        factor = tracker.get_momentum_factor()
        assert factor <= 1.5

    def test_reset_clears_state(self) -> None:
        tracker = MomentumTracker()
        for _ in range(5):
            tracker.update(1.0)
        tracker.reset()
        assert tracker.streak_count == 0
        assert tracker.streak_sign == 0

    def test_load_restores_state(self) -> None:
        tracker = MomentumTracker()
        tracker.load(streak_count=5, streak_sign=1)
        assert tracker.streak_count == 5
        assert tracker.streak_sign == 1

    def test_cooldown_after_streak_break(self) -> None:
        config = MomentumConfig(streak_threshold=3, cooldown_steps=2)
        tracker = MomentumTracker(config)
        for _ in range(4):
            tracker.update(1.0)
        # Break streak
        tracker.update(-1.0)
        # Should be in cooldown
        factor = tracker.get_momentum_factor()
        assert factor == 0.0

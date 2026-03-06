"""Tests for tonic baseline, phasic processor, and dual-mode integration."""

from dopamine_core.config import PhasicConfig, TonicConfig
from dopamine_core.reward.dual_mode import DualModeReward
from dopamine_core.reward.phasic import PhasicProcessor
from dopamine_core.reward.tonic import TonicBaseline
from dopamine_core.types import RPEResult


class TestTonicBaseline:
    def test_initial_level_is_zero(self) -> None:
        tonic = TonicBaseline()
        assert tonic.level == 0.0

    def test_positive_rpe_raises_baseline(self) -> None:
        tonic = TonicBaseline()
        for _ in range(10):
            tonic.update(0.5)
        assert tonic.level > 0.0

    def test_negative_rpe_lowers_baseline(self) -> None:
        tonic = TonicBaseline()
        for _ in range(10):
            tonic.update(-0.5)
        assert tonic.level < 0.0

    def test_baseline_clamped_to_max(self) -> None:
        config = TonicConfig(learning_rate=0.5, max_level=1.0, decay_rate=0.98)
        tonic = TonicBaseline(config)
        for _ in range(100):
            tonic.update(10.0)
        assert tonic.level <= 1.0

    def test_baseline_clamped_to_negative_max(self) -> None:
        config = TonicConfig(learning_rate=0.5, max_level=1.0, decay_rate=0.98)
        tonic = TonicBaseline(config)
        for _ in range(100):
            tonic.update(-10.0)
        assert tonic.level >= -1.0

    def test_reset_restores_initial(self) -> None:
        tonic = TonicBaseline()
        tonic.update(0.5)
        tonic.reset()
        assert tonic.level == 0.0
        assert tonic.step_count == 0

    def test_load_restores_state(self) -> None:
        tonic = TonicBaseline()
        tonic.load(baseline=0.42, step_count=10)
        assert tonic.level == 0.42
        assert tonic.step_count == 10

    def test_decay_rate_brings_baseline_toward_zero(self) -> None:
        config = TonicConfig(decay_rate=0.9, learning_rate=0.05)
        tonic = TonicBaseline(config)
        tonic.load(baseline=1.0, step_count=0)
        # With no input, decay should bring it down
        for _ in range(20):
            tonic.update(0.0)
        assert tonic.level < 0.5


class TestPhasicProcessor:
    def test_positive_rpe_produces_positive_response(self) -> None:
        phasic = PhasicProcessor()
        result = phasic.process(rpe_error=0.5, raw_error=0.5)
        assert result > 0

    def test_negative_rpe_produces_negative_response(self) -> None:
        phasic = PhasicProcessor()
        result = phasic.process(rpe_error=-0.935, raw_error=-0.5)
        assert result < 0

    def test_dip_scale_amplifies_negative(self) -> None:
        config = PhasicConfig(dip_scale=2.0, burst_scale=1.0)
        phasic = PhasicProcessor(config)
        result = phasic.process(rpe_error=-0.5, raw_error=-0.5)
        assert result == -1.0  # -0.5 * 2.0

    def test_burst_scale_amplifies_positive(self) -> None:
        config = PhasicConfig(burst_scale=2.0, dip_scale=1.87)
        phasic = PhasicProcessor(config)
        result = phasic.process(rpe_error=0.5, raw_error=0.5)
        assert result == 1.0  # 0.5 * 2.0

    def test_decayed_influence_diminishes(self) -> None:
        phasic = PhasicProcessor()
        phasic.process(rpe_error=1.0, raw_error=1.0)
        immediate = phasic.get_decayed_influence()

        # Simulate time passing by processing small signals
        for _ in range(10):
            phasic.process(rpe_error=0.0, raw_error=0.0)

        later = phasic.get_decayed_influence()
        assert abs(later) < abs(immediate)

    def test_reset_clears_state(self) -> None:
        phasic = PhasicProcessor()
        phasic.process(rpe_error=0.5, raw_error=0.5)
        phasic.reset()
        assert phasic.current_response == 0.0
        assert phasic.get_decayed_influence() == 0.0

    def test_history_tracking(self) -> None:
        phasic = PhasicProcessor()
        phasic.process(rpe_error=0.5, raw_error=0.5)
        phasic.process(rpe_error=-0.3, raw_error=-0.3)
        history = phasic.get_history()
        assert len(history) == 2

    def test_load_history(self) -> None:
        phasic = PhasicProcessor()
        phasic.load_history([(0.5, 0), (-0.3, 1)])
        assert phasic.current_response == 0.5
        assert len(phasic.get_history()) == 2


class TestDualModeReward:
    def _make_rpe(self, error: float, raw_error: float) -> RPEResult:
        return RPEResult(
            prediction=0.0,
            actual=0.5,
            error=error,
            raw_error=raw_error,
            surprise=abs(raw_error),
        )

    def test_positive_rpe_produces_positive_composite(self) -> None:
        dual = DualModeReward()
        result = dual.process(self._make_rpe(0.5, 0.5))
        assert result > 0

    def test_negative_rpe_produces_negative_composite(self) -> None:
        dual = DualModeReward()
        result = dual.process(self._make_rpe(-0.935, -0.5))
        assert result < 0

    def test_tonic_contributes_to_composite(self) -> None:
        dual = DualModeReward()
        # Build up a positive tonic baseline
        for _ in range(20):
            dual.process(self._make_rpe(0.5, 0.5))

        tonic_level = dual.tonic.level
        assert tonic_level > 0

        # Now send a neutral signal — tonic should keep composite positive
        result = dual.process(self._make_rpe(0.0, 0.0))
        assert result > 0  # tonic contribution keeps it positive

    def test_reset_clears_both(self) -> None:
        dual = DualModeReward()
        dual.process(self._make_rpe(0.5, 0.5))
        dual.reset()
        assert dual.tonic.level == 0.0
        assert dual.phasic.current_response == 0.0

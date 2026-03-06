"""Tests for distributional reward coding."""

from dopamine_core.config import DistributionalConfig
from dopamine_core.distributional.channels import DistributionalChannels, RewardChannel
from dopamine_core.distributional.coding import DistributionalCoding


class TestRewardChannel:
    def test_initial_expectation_is_zero(self) -> None:
        ch = RewardChannel(tau=0.5, learning_rate=0.05)
        assert ch.expectation == 0.0

    def test_positive_outcome_raises_expectation(self) -> None:
        ch = RewardChannel(tau=0.5, learning_rate=0.1)
        for _ in range(10):
            ch.update(1.0)
        assert ch.expectation > 0.0

    def test_high_tau_learns_more_from_positive(self) -> None:
        optimistic = RewardChannel(tau=0.9, learning_rate=0.1)
        pessimistic = RewardChannel(tau=0.1, learning_rate=0.1)
        for _ in range(10):
            optimistic.update(1.0)
            pessimistic.update(1.0)
        assert optimistic.expectation > pessimistic.expectation

    def test_low_tau_learns_more_from_negative(self) -> None:
        optimistic = RewardChannel(tau=0.9, learning_rate=0.1)
        pessimistic = RewardChannel(tau=0.1, learning_rate=0.1)
        for _ in range(10):
            optimistic.update(-1.0)
            pessimistic.update(-1.0)
        assert pessimistic.expectation < optimistic.expectation

    def test_load_sets_expectation(self) -> None:
        ch = RewardChannel(tau=0.5, learning_rate=0.1)
        ch.load(0.42)
        assert ch.expectation == 0.42


class TestDistributionalChannels:
    def test_default_creates_5_channels(self) -> None:
        dc = DistributionalChannels()
        assert dc.num_channels == 5

    def test_custom_channel_count(self) -> None:
        config = DistributionalConfig(num_channels=7)
        dc = DistributionalChannels(config)
        assert dc.num_channels == 7

    def test_update_returns_per_channel_errors(self) -> None:
        dc = DistributionalChannels()
        errors = dc.update(0.5)
        assert len(errors) == 5

    def test_expectations_diverge_after_updates(self) -> None:
        dc = DistributionalChannels()
        for _ in range(20):
            dc.update(0.8)
        exps = dc.expectations
        # Optimistic channels (high tau) should have higher expectations
        assert exps[-1] > exps[0]

    def test_spread_increases_with_varied_outcomes(self) -> None:
        dc = DistributionalChannels()
        initial_spread = dc.get_spread()
        for i in range(20):
            dc.update(1.0 if i % 2 == 0 else -1.0)
        assert dc.get_spread() > initial_spread

    def test_mean_expectation_tracks_outcomes(self) -> None:
        dc = DistributionalChannels()
        for _ in range(50):
            dc.update(0.8)
        assert dc.get_mean_expectation() > 0.3

    def test_skew_bounded(self) -> None:
        dc = DistributionalChannels()
        for _ in range(50):
            dc.update(1.0)
        skew = dc.get_skew()
        assert -1.0 <= skew <= 1.0

    def test_reset_restores_initial(self) -> None:
        dc = DistributionalChannels()
        dc.update(1.0)
        dc.reset()
        assert all(e == 0.0 for e in dc.expectations)

    def test_load_expectations(self) -> None:
        dc = DistributionalChannels()
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        dc.load_expectations(values)
        assert dc.expectations == values


class TestDistributionalCoding:
    def test_neutral_state_gives_near_zero_risk(self) -> None:
        dc = DistributionalChannels()
        coding = DistributionalCoding(dc)
        assert abs(coding.get_risk_score()) < 0.01

    def test_uncertainty_near_zero_initially(self) -> None:
        dc = DistributionalChannels()
        coding = DistributionalCoding(dc)
        assert coding.get_uncertainty() < 0.01

    def test_uncertainty_rises_with_varied_outcomes(self) -> None:
        dc = DistributionalChannels()
        coding = DistributionalCoding(dc)
        for i in range(30):
            dc.update(1.0 if i % 2 == 0 else -1.0)
        assert coding.get_uncertainty() > 0.1

    def test_risk_score_bounded(self) -> None:
        dc = DistributionalChannels()
        coding = DistributionalCoding(dc)
        for _ in range(100):
            dc.update(1.0)
        assert -1.0 <= coding.get_risk_score() <= 1.0

    def test_uncertainty_bounded(self) -> None:
        dc = DistributionalChannels()
        coding = DistributionalCoding(dc)
        for _ in range(100):
            dc.update(10.0)
        assert 0.0 <= coding.get_uncertainty() <= 1.0

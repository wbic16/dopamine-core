"""Tests for signal extraction and RPE computation."""

from dopamine_core.signals.extractor import SignalExtractor
from dopamine_core.signals.rpe import RPECalculator


class TestSignalExtractor:
    def setup_method(self) -> None:
        self.extractor = SignalExtractor()

    def test_empty_text_returns_zeros(self) -> None:
        result = self.extractor.extract("")
        assert result.confidence == 0.0
        assert result.risk_framing == 0.0
        assert result.deliberation_depth == 0.0
        assert result.temporal_references == 0.0

    def test_high_confidence_text(self) -> None:
        text = "I'm very confident that BTC will rise. Confidence: high. This will definitely happen."
        result = self.extractor.extract(text)
        assert result.confidence > 0.5

    def test_low_confidence_text(self) -> None:
        text = "Maybe BTC will go up, but I'm not sure. It's hard to say. Perhaps it could drop."
        result = self.extractor.extract(text)
        assert result.confidence < -0.3

    def test_risk_seeking_text(self) -> None:
        text = "Big position here, aggressive strategy. All-in on this trade. High reward potential."
        result = self.extractor.extract(text)
        assert result.risk_framing > 0.3

    def test_risk_averse_text(self) -> None:
        text = "Setting a stop-loss here. Conservative approach with hedging. Small position due to downside risk."
        result = self.extractor.extract(text)
        assert result.risk_framing < -0.3

    def test_deliberation_depth(self) -> None:
        shallow = "Buy BTC."
        deep = (
            "Because institutional buying increased, and furthermore the RSI shows oversold. "
            "However, the 200-day moving average at $45,000 suggests resistance. "
            "Therefore, I recommend a 30% position with a stop-loss at $42,500. "
            "Additionally, the macro environment with 5.25% rates creates headwinds. "
            "Given that volume is declining, the setup has a 65% probability."
        )
        shallow_result = self.extractor.extract(shallow)
        deep_result = self.extractor.extract(deep)
        assert deep_result.deliberation_depth > shallow_result.deliberation_depth

    def test_temporal_references(self) -> None:
        no_refs = "BTC will rise because of supply dynamics."
        with_refs = (
            "Last time we saw this pattern, the market dropped 20%. "
            "Based on my past trades, I've learned that historically "
            "these setups have a 60% win rate. Recently performance has been mixed."
        )
        no_refs_result = self.extractor.extract(no_refs)
        refs_result = self.extractor.extract(with_refs)
        assert refs_result.temporal_references > no_refs_result.temporal_references

    def test_confidence_clamped_to_bounds(self) -> None:
        # Even with many confident signals, should be clamped to [-1, 1]
        text = (
            "I'm very confident. Definitely. Strongly believe. Confidence: high. "
            "100% sure. Clearly. I am certain. Strongly recommend."
        )
        result = self.extractor.extract(text)
        assert -1.0 <= result.confidence <= 1.0


class TestRPECalculator:
    def setup_method(self) -> None:
        self.rpe = RPECalculator()

    def test_high_confidence_wrong_large_negative(self) -> None:
        # outcome=0 (loss), confidence=1.0 (max confident)
        result = self.rpe.compute(outcome=0.0, confidence=1.0)
        assert result.error < -0.5

    def test_low_confidence_right_moderate_positive(self) -> None:
        # outcome=1 (win), confidence=-1.0 (very uncertain)
        result = self.rpe.compute(outcome=1.0, confidence=-1.0)
        assert result.error > 0

    def test_loss_aversion_amplifies_negatives(self) -> None:
        # Same magnitude, different sign
        result_pos = self.rpe.compute(outcome=1.0, confidence=0.0)
        result_neg = self.rpe.compute(outcome=0.0, confidence=0.0)
        # Negative should be amplified by 1.87x
        assert abs(result_neg.error) > abs(result_pos.error)

    def test_loss_aversion_multiplier(self) -> None:
        result = self.rpe.compute(outcome=0.0, confidence=0.0)
        # With confidence=0, conf_normalized=0.5
        # RPE = 0*(1-0.5) + (1-0)*(-0.5) = -0.5
        # After loss aversion: -0.5 * 1.87 = -0.935
        assert abs(result.error - (-0.5 * 1.87)) < 0.01

    def test_neutral_outcome_neutral_confidence(self) -> None:
        result = self.rpe.compute(outcome=0.5, confidence=0.0)
        # outcome=0.5, conf_normalized=0.5
        # RPE = 0.5*0.5 + 0.5*(-0.5) = 0.25 - 0.25 = 0
        assert abs(result.raw_error) < 0.01

    def test_surprise_is_bounded(self) -> None:
        result = self.rpe.compute(outcome=1.0, confidence=1.0, baseline=0.0)
        assert result.surprise <= 10.0

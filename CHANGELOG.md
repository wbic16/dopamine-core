# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-03-06

### Added

- Core `DopamineEngine` orchestrator with `inject_context()` and `update()` API
- Confidence-weighted Reward Prediction Error (RPE) computation with 1.87x loss aversion
- Chain-of-Thought behavioral signal extraction (confidence, risk framing, deliberation depth, temporal references)
- Subliminal context injection with naturalistic environmental templates
- Tonic baseline adaptation via exponential moving average
- Win/loss streak tracking with configurable momentum
- Full Pydantic configuration schema with sensible defaults
- Safety signal clamping to bounded range
- Engine state serialization and restoration
- Comprehensive test suite (29 tests)

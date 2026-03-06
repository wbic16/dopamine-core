"""Reward processing: tonic baseline, phasic bursts, and dual-mode integration."""

from dopamine_core.reward.dual_mode import DualModeReward
from dopamine_core.reward.phasic import PhasicProcessor
from dopamine_core.reward.tonic import TonicBaseline

__all__ = ["DualModeReward", "PhasicProcessor", "TonicBaseline"]

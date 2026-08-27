"""AlignPy: lightweight, modular LLM alignment on top of Hugging Face TRL.

Public API::

    from alignpy import AlignmentConfig, DPOTrainer, GRPOTrainer, SFTTrainer

Config classes import instantly (pydantic only); trainer classes are resolved
lazily on first attribute access, so config authoring and validation work in
environments without torch/trl installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alignpy.configs import (
    AlignmentConfig,
    DataConfig,
    DPOParams,
    GRPOParams,
    ModelConfig,
    PeftConfig,
    SFTParams,
    TrainingConfig,
)
from alignpy.rewards import RewardFunction, register_reward

if TYPE_CHECKING:
    from alignpy.trainers import (
        DPOTrainer,
        GRPOTrainer,
        RewardLogHook,
        RewardMetricsCallback,
        SFTTrainer,
        build_trainer,
    )

__version__ = "0.1.0"

__all__ = [
    "AlignmentConfig",
    "DataConfig",
    "DPOParams",
    "DPOTrainer",
    "GRPOParams",
    "GRPOTrainer",
    "ModelConfig",
    "PeftConfig",
    "RewardFunction",
    "RewardLogHook",
    "RewardMetricsCallback",
    "SFTParams",
    "SFTTrainer",
    "TrainingConfig",
    "build_trainer",
    "register_reward",
    "__version__",
]

_TRAINER_EXPORTS = {
    "DPOTrainer",
    "GRPOTrainer",
    "RewardLogHook",
    "RewardMetricsCallback",
    "SFTTrainer",
    "build_trainer",
}


def __getattr__(name: str):
    if name in _TRAINER_EXPORTS:
        import alignpy.trainers as trainers

        return getattr(trainers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""AlignPy trainers: config-driven wrappers around TRL's alignment trainers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alignpy.configs import AlignmentConfig

if TYPE_CHECKING:
    from alignpy.trainers.base import BaseAlignmentTrainer

__all__ = [
    "BaseAlignmentTrainer",
    "DPOTrainer",
    "GRPOTrainer",
    "RewardLogHook",
    "RewardMetricsCallback",
    "SFTTrainer",
    "build_trainer",
]

# Class names are resolved lazily (PEP 562) so that importing alignpy.trainers
# for build_trainer's signature does not require torch/trl to be installed.
_LAZY = {
    "BaseAlignmentTrainer": ("alignpy.trainers.base", "BaseAlignmentTrainer"),
    "RewardLogHook": ("alignpy.trainers.base", "RewardLogHook"),
    "RewardMetricsCallback": ("alignpy.trainers.base", "RewardMetricsCallback"),
    "SFTTrainer": ("alignpy.trainers.sft", "SFTTrainer"),
    "DPOTrainer": ("alignpy.trainers.dpo", "DPOTrainer"),
    "GRPOTrainer": ("alignpy.trainers.grpo", "GRPOTrainer"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)


def build_trainer(config: AlignmentConfig, *args, **kwargs) -> BaseAlignmentTrainer:
    """Instantiate the trainer matching ``config.method``.

    Extra positional/keyword arguments (datasets, reward_hooks, callbacks, …)
    are forwarded to the trainer's constructor. This is the dispatch point the
    CLI uses, and the recommended entry when the method is chosen at runtime.
    """
    trainer_cls = __getattr__(
        {"sft": "SFTTrainer", "dpo": "DPOTrainer", "grpo": "GRPOTrainer"}[config.method]
    )
    return trainer_cls(config, *args, **kwargs)

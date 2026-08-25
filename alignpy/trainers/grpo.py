"""AlignPy's Group Relative Policy Optimization trainer.

Wraps :class:`trl.GRPOTrainer` for reasoning-style optimization: sample a
group of completions per prompt, score them with the reward pipeline, and
update the policy on group-normalized advantages.

The reward pipeline mixes three sources, in one list:

1. Registered custom functions — names from ``config.grpo.reward_funcs``
   resolved via :mod:`alignpy.rewards`.
2. Reward models — unregistered names are passed through to TRL as Hub model
   ids for sequence-classification reward models.
3. Ad-hoc callables — passed directly as ``reward_funcs=`` to this trainer,
   appended after the config-declared ones.

Reward telemetry (``reward``, ``reward_std``, per-function means) reaches any
configured :class:`~alignpy.trainers.base.RewardLogHook` every logging step.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

from trl import GRPOConfig as TRLGRPOConfig
from trl import GRPOTrainer as TRLGRPOTrainer

from alignpy.configs import AlignmentConfig, AlignmentMethod
from alignpy.rewards import RewardFunction, resolve_reward
from alignpy.trainers.base import BaseAlignmentTrainer

__all__ = ["GRPOTrainer"]


class GRPOTrainer(BaseAlignmentTrainer):
    """Optimize a policy against custom rewards with GRPO (``method: "grpo"``).

    Parameters beyond the shared ones (see :class:`BaseAlignmentTrainer`):

    reward_funcs:
        Extra ad-hoc reward callables appended after those resolved from
        ``config.grpo.reward_funcs``. At least one reward source must exist
        between the two.
    """

    method: ClassVar[AlignmentMethod] = "grpo"

    def __init__(
        self,
        config: AlignmentConfig,
        *args,
        reward_funcs: Sequence[RewardFunction] = (),
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)
        self._extra_reward_funcs = list(reward_funcs)

    def _build_trainer(self) -> TRLGRPOTrainer:
        data = self.config.data
        grpo = self.config.grpo
        assert grpo is not None  # guaranteed by AlignmentConfig validation

        reward_funcs = [resolve_reward(name) for name in grpo.reward_funcs]
        reward_funcs += self._extra_reward_funcs
        if not reward_funcs:
            raise ValueError(
                "GRPO needs at least one reward: set grpo.reward_funcs in the config "
                "(registered names or reward-model ids) or pass reward_funcs= callables."
            )
        reward_weights = grpo.reward_weights
        if reward_weights is not None and self._extra_reward_funcs:
            # Config weights only cover config-declared functions; ad-hoc ones get 1.0.
            reward_weights = reward_weights + [1.0] * len(self._extra_reward_funcs)

        train_dataset, eval_dataset = self._load_datasets(
            column_mapping={data.prompt_column: "prompt"}
        )

        args = TRLGRPOConfig(
            **self._base_training_kwargs(),
            num_generations=grpo.num_generations,
            max_prompt_length=grpo.max_prompt_length,
            max_completion_length=grpo.max_completion_length,
            temperature=grpo.temperature,
            top_p=grpo.top_p,
            beta=grpo.beta,
            epsilon=grpo.epsilon,
            reward_weights=reward_weights,
            scale_rewards=grpo.scale_rewards,
        )

        return TRLGRPOTrainer(
            model=self._load_model(),
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self._load_tokenizer(),
            peft_config=self._peft_config(),
            callbacks=self._callbacks() or None,
        )

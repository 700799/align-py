"""AlignPy's Direct Preference Optimization trainer.

A clean, config-driven wrapper around :class:`trl.DPOTrainer`. You provide a
validated :class:`~alignpy.configs.AlignmentConfig` (``method: "dpo"``) and,
optionally, chosen/rejected preference datasets and *reward logging hooks*;
AlignPy handles model/tokenizer loading, PEFT wiring, dataset column mapping,
and the translation into ``trl.DPOConfig``.

Reward hooks
------------
DPO's loss induces an *implicit* reward ``beta * log(pi/pi_ref)``; the gap
between chosen and rejected responses (``rewards/margins``) is the single best
live indicator that preference alignment is actually happening. TRL computes
these statistics every logging step, and any :class:`RewardLogHook` passed to
this trainer receives them as ``(metrics, step)`` — use hooks to stream
margins to a dashboard, assert training health, or trigger early stopping
logic of your own.

Example
-------
>>> from alignpy import AlignmentConfig, DPOTrainer
>>> config = AlignmentConfig.from_yaml("dpo_config.yaml")
>>> trainer = DPOTrainer(config, reward_hooks=[lambda m, s: print(s, m)])
>>> trainer.train()
>>> trainer.save()
"""

from __future__ import annotations

from typing import ClassVar

from trl import DPOConfig as TRLDPOConfig
from trl import DPOTrainer as TRLDPOTrainer

from alignpy.configs import AlignmentMethod
from alignpy.trainers.base import (
    BaseAlignmentTrainer,
    RewardLogHook,
    RewardMetricsCallback,
)

__all__ = ["DPOTrainer", "RewardLogHook", "RewardMetricsCallback"]


class DPOTrainer(BaseAlignmentTrainer):
    """Align a post-SFT model on chosen/rejected preference pairs via DPO.

    Parameters
    ----------
    config:
        A validated :class:`~alignpy.configs.AlignmentConfig` with
        ``method: "dpo"``. Its ``dpo`` block (beta, loss_type, length caps, …)
        is translated into ``trl.DPOConfig`` at build time.
    train_dataset / eval_dataset:
        Optional pre-loaded ``datasets.Dataset`` objects with ``prompt`` /
        ``chosen`` / ``rejected`` columns (differently-named columns are
        remapped per ``config.data``). When omitted, datasets are loaded from
        ``config.data`` on first use.
    reward_hooks:
        Callables receiving TRL's implicit-reward telemetry
        (``rewards/chosen|rejected|margins|accuracies``) at every logging
        step — see :class:`RewardLogHook`.
    callbacks:
        Extra raw ``transformers.TrainerCallback`` instances, forwarded
        untouched for anything beyond reward logging.

    Everything heavy is lazy: models and datasets load on the first call to
    :meth:`train` / :meth:`evaluate` (or :attr:`trainer` access), never in
    ``__init__``.
    """

    method: ClassVar[AlignmentMethod] = "dpo"

    def _build_trainer(self) -> TRLDPOTrainer:
        """Materialize models and datasets, and wire up ``trl.DPOTrainer``."""
        data = self.config.data
        dpo = self.config.dpo
        assert dpo is not None  # guaranteed by AlignmentConfig validation

        train_dataset, eval_dataset = self._load_datasets(
            column_mapping={
                data.prompt_column: "prompt",
                data.chosen_column: "chosen",
                data.rejected_column: "rejected",
            }
        )

        # TRL >= 1.0 enforces one combined sequence cap; AlignPy's separate
        # prompt/completion caps are validated upstream and folded into it.
        args = TRLDPOConfig(
            **self._base_training_kwargs(),
            beta=dpo.beta,
            loss_type=dpo.loss_type,
            label_smoothing=dpo.label_smoothing,
            max_length=dpo.max_length,
            truncation_mode=dpo.truncation_mode,
        )

        return TRLDPOTrainer(
            model=self._load_model(),
            ref_model=self._load_ref_model(),
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self._load_tokenizer(),
            peft_config=self._peft_config(),
            callbacks=self._callbacks() or None,
        )

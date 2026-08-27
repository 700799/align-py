"""AlignPy's supervised fine-tuning trainer.

Wraps :class:`trl.SFTTrainer` to produce the instruction-following baseline
that DPO/GRPO alignment starts from. Datasets may provide either a formatted
``text`` column (remapped per ``config.data.text_column``) or a conversational
``messages`` column, which TRL formats with the tokenizer's chat template.
"""

from __future__ import annotations

from typing import ClassVar

from trl import SFTConfig as TRLSFTConfig
from trl import SFTTrainer as TRLSFTTrainer

from alignpy.configs import AlignmentMethod
from alignpy.trainers.base import BaseAlignmentTrainer

__all__ = ["SFTTrainer"]


class SFTTrainer(BaseAlignmentTrainer):
    """Instruction-tune a base model on demonstrations (``method: "sft"``).

    Accepts the same constructor arguments as every AlignPy trainer
    (``config``, optional datasets, ``reward_hooks``, ``callbacks``); reward
    hooks are simply never fired here since SFT produces no reward telemetry.
    """

    method: ClassVar[AlignmentMethod] = "sft"

    def _build_trainer(self) -> TRLSFTTrainer:
        data = self.config.data
        sft = self.config.sft
        assert sft is not None  # guaranteed by AlignmentConfig validation

        train_dataset, eval_dataset = self._load_datasets(
            column_mapping={data.text_column: "text"}
        )

        args = TRLSFTConfig(
            **self._base_training_kwargs(),
            max_length=sft.max_length,
            packing=sft.packing,
        )

        return TRLSFTTrainer(
            model=self._load_model(),
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self._load_tokenizer(),
            peft_config=self._peft_config(),
            callbacks=self._callbacks() or None,
        )

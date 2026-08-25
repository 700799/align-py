"""Shared plumbing for AlignPy trainers.

Every AlignPy trainer is a thin, config-driven wrapper around a TRL trainer:
:class:`BaseAlignmentTrainer` owns the parts that are identical across
methods — loading the tokenizer/model from :class:`~alignpy.configs.ModelConfig`,
building an optional ``peft.LoraConfig``, loading/renaming datasets, and the
``train`` / ``evaluate`` / ``save`` lifecycle — while each subclass implements
``_build_trainer`` to translate its parameter block into the matching TRL
trainer.

This module also defines the *reward logging hook* contract shared by DPO and
GRPO: TRL emits reward telemetry into the trainer logs (implicit reward
margins for DPO, per-function rewards for GRPO), and
:class:`RewardMetricsCallback` filters those entries out of each log event and
dispatches them to user hooks.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Protocol, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TrainerCallback,
)

from alignpy.configs import AlignmentConfig, AlignmentMethod

if TYPE_CHECKING:
    from datasets import Dataset
    from peft import LoraConfig

__all__ = ["BaseAlignmentTrainer", "RewardLogHook", "RewardMetricsCallback"]

#: Filename of the AlignPy config snapshot written next to saved models, so
#: ``alignpy eval --model <dir>`` can reconstruct the run without extra flags.
CONFIG_SNAPSHOT_FILENAME = "alignpy_config.yaml"

#: Log keys (by prefix or exact name) that constitute reward telemetry.
_REWARD_KEY_PREFIXES = ("rewards/", "eval_rewards/")
_REWARD_KEY_EXACT = ("reward", "reward_std", "eval_reward", "eval_reward_std")


class RewardLogHook(Protocol):
    """A callable invoked with reward telemetry each time the trainer logs.

    ``metrics`` maps TRL reward keys to floats — for DPO e.g.
    ``rewards/chosen``, ``rewards/rejected``, ``rewards/margins``,
    ``rewards/accuracies``; for GRPO e.g. ``reward``, ``reward_std``, and
    ``rewards/<func_name>/mean``. ``step`` is the global optimizer step.
    Hooks are observers: they must not mutate training state.
    """

    def __call__(self, metrics: Mapping[str, float], step: int) -> None:
        ...


class RewardMetricsCallback(TrainerCallback):
    """Bridges TRL's log stream to AlignPy reward hooks.

    TRL trainers report reward statistics through the standard
    ``transformers`` logging pipeline; this callback intercepts each log
    event, keeps only reward-related keys, and fans them out to the
    registered :class:`RewardLogHook` callables.
    """

    def __init__(self, hooks: Sequence[RewardLogHook]):
        self._hooks = list(hooks)

    def on_log(self, args, state, control, logs: dict[str, Any] | None = None, **kwargs):
        if not logs or not self._hooks:
            return
        metrics = {
            key: float(value)
            for key, value in logs.items()
            if isinstance(value, (int, float))
            and (key.startswith(_REWARD_KEY_PREFIXES) or key in _REWARD_KEY_EXACT)
        }
        if metrics:
            for hook in self._hooks:
                hook(metrics, state.global_step)


class BaseAlignmentTrainer(abc.ABC):
    """Config-driven lifecycle shared by :class:`SFTTrainer`, :class:`DPOTrainer`,
    and :class:`GRPOTrainer`.

    Construction is cheap and validates intent only; models, tokenizers, and
    datasets are materialized lazily on first use of :attr:`trainer` (or
    :meth:`train`), so a misconfigured run fails before any download starts.
    """

    #: The ``AlignmentConfig.method`` value this trainer implements.
    method: ClassVar[AlignmentMethod]

    def __init__(
        self,
        config: AlignmentConfig,
        train_dataset: "Dataset | None" = None,
        eval_dataset: "Dataset | None" = None,
        *,
        reward_hooks: Sequence[RewardLogHook] = (),
        callbacks: Sequence[TrainerCallback] = (),
    ) -> None:
        if config.method != self.method:
            raise ValueError(
                f"{type(self).__name__} implements method={self.method!r} but the config "
                f"declares method={config.method!r}. Use alignpy.trainers.build_trainer() "
                "to dispatch on the config, or fix the config's 'method' field."
            )
        self.config = config
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._reward_hooks = list(reward_hooks)
        self._extra_callbacks = list(callbacks)
        self._trainer = None  # underlying TRL trainer, built lazily

    # ------------------------------------------------------------------ #
    # Lazy build
    # ------------------------------------------------------------------ #

    @property
    def trainer(self):
        """The underlying TRL trainer, built on first access."""
        if self._trainer is None:
            self._trainer = self._build_trainer()
        return self._trainer

    @abc.abstractmethod
    def _build_trainer(self):
        """Translate ``self.config`` into a fully-wired TRL trainer instance."""

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def train(self):
        """Run the alignment pass and return the ``transformers`` train output."""
        return self.trainer.train()

    def evaluate(self) -> dict[str, float]:
        """Evaluate on the eval dataset (requires one to be configured)."""
        if self._eval_dataset is None and self.config.data.eval_split is None:
            raise ValueError(
                "No eval data: pass eval_dataset= or set data.eval_split in the config."
            )
        return self.trainer.evaluate()

    def save(self, output_dir: str | Path | None = None) -> Path:
        """Save model, tokenizer, and a config snapshot; returns the directory.

        The snapshot (``alignpy_config.yaml``) makes saved models
        self-describing: ``alignpy eval --model <dir>`` reads it to rebuild
        the evaluation pipeline with no further flags.
        """
        output_dir = Path(output_dir or self.config.train.output_dir)
        self.trainer.save_model(str(output_dir))
        if self.trainer.processing_class is not None:
            self.trainer.processing_class.save_pretrained(str(output_dir))
        self.config.to_yaml(output_dir / CONFIG_SNAPSHOT_FILENAME)
        return output_dir

    # ------------------------------------------------------------------ #
    # Shared component loaders (used by subclasses inside _build_trainer)
    # ------------------------------------------------------------------ #

    def _load_tokenizer(self) -> Any:
        cfg = self.config.model
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=cfg.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self, name_or_path: str | None = None) -> PreTrainedModel:
        cfg = self.config.model
        dtype = cfg.torch_dtype if cfg.torch_dtype == "auto" else getattr(torch, cfg.torch_dtype)
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": cfg.trust_remote_code,
        }
        if cfg.attn_implementation is not None:
            kwargs["attn_implementation"] = cfg.attn_implementation
        return AutoModelForCausalLM.from_pretrained(
            name_or_path or cfg.model_name_or_path, **kwargs
        )

    def _load_ref_model(self) -> PreTrainedModel | None:
        """Explicit reference model, or None to let TRL derive one.

        Under PEFT no separate reference is needed — TRL evaluates the
        reference policy by disabling the adapters — so an explicit ref is
        only loaded for full-parameter runs that configured one.
        """
        cfg = self.config.model
        if self.config.peft.enabled or cfg.ref_model_name_or_path is None:
            return None
        return self._load_model(cfg.ref_model_name_or_path)

    def _peft_config(self) -> "LoraConfig | None":
        if not self.config.peft.enabled:
            return None
        from peft import LoraConfig

        peft = self.config.peft
        return LoraConfig(
            r=peft.r,
            lora_alpha=peft.lora_alpha,
            lora_dropout=peft.lora_dropout,
            target_modules=peft.target_modules,
            task_type="CAUSAL_LM",
        )

    def _load_datasets(self, column_mapping: Mapping[str, str]) -> tuple["Dataset", "Dataset | None"]:
        """Return (train, eval) datasets, loading from config when not injected.

        ``column_mapping`` maps configured column names to the canonical names
        the TRL trainer expects (e.g. ``{data.chosen_column: "chosen"}``);
        identity entries are skipped.
        """
        from datasets import load_dataset

        data = self.config.data
        train = self._train_dataset
        if train is None:
            train = load_dataset(data.dataset_name_or_path, split=data.split)
        eval_ds = self._eval_dataset
        if eval_ds is None and data.eval_split is not None:
            eval_ds = load_dataset(data.dataset_name_or_path, split=data.eval_split)

        def prepare(ds: "Dataset") -> "Dataset":
            renames = {
                src: dst
                for src, dst in column_mapping.items()
                if src != dst and src in ds.column_names
            }
            if renames:
                ds = ds.rename_columns(renames)
            if data.max_samples is not None:
                ds = ds.select(range(min(data.max_samples, len(ds))))
            return ds

        return prepare(train), (prepare(eval_ds) if eval_ds is not None else None)

    def _base_training_kwargs(self) -> dict[str, Any]:
        """TrainingConfig fields shared by every TRL config class."""
        t = self.config.train
        return {
            "output_dir": t.output_dir,
            "learning_rate": t.learning_rate,
            "per_device_train_batch_size": t.per_device_train_batch_size,
            "per_device_eval_batch_size": t.per_device_eval_batch_size,
            "gradient_accumulation_steps": t.gradient_accumulation_steps,
            "num_train_epochs": t.num_train_epochs,
            "max_steps": t.max_steps,
            "lr_scheduler_type": t.lr_scheduler_type,
            "warmup_ratio": t.warmup_ratio,
            "logging_steps": t.logging_steps,
            "save_strategy": t.save_strategy,
            "save_steps": t.save_steps,
            "eval_strategy": t.eval_strategy,
            "eval_steps": t.eval_steps,
            "bf16": t.bf16,
            "fp16": t.fp16,
            "gradient_checkpointing": t.gradient_checkpointing,
            "seed": t.seed,
            "report_to": t.report_to,
        }

    def _callbacks(self) -> list[TrainerCallback]:
        """User callbacks plus the reward-hook bridge (when hooks are set)."""
        callbacks = list(self._extra_callbacks)
        if self._reward_hooks:
            callbacks.append(RewardMetricsCallback(self._reward_hooks))
        return callbacks

"""Pydantic V2 configuration schemas for AlignPy.

Everything a user can tune lives here, in one strictly-validated tree rooted at
:class:`AlignmentConfig`. The schema is intentionally decoupled from TRL's own
config dataclasses: AlignPy validates early (at YAML-load / construction time,
before any model weights are touched) and each trainer translates the validated
tree into the corresponding TRL config (``DPOConfig``, ``GRPOConfig``,
``SFTConfig``) at build time.

Layout::

    AlignmentConfig
    ├── method: "sft" | "dpo" | "grpo"   # which alignment algorithm to run
    ├── model:  ModelConfig              # policy (and optional reference) model
    ├── data:   DataConfig               # dataset location + column mapping
    ├── train:  TrainingConfig           # optimizer / schedule / logging
    ├── peft:   PeftConfig               # optional LoRA adaptation
    └── sft | dpo | grpo                 # method-specific hyperparameters

Only the parameter block matching ``method`` is required; it is auto-created
with defaults when omitted. Unknown keys are rejected everywhere
(``extra="forbid"``) so a typo like ``betta: 0.1`` fails loudly instead of
silently training with defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AlignmentConfig",
    "AlignmentMethod",
    "DataConfig",
    "DPOLossType",
    "DPOParams",
    "GRPOParams",
    "ModelConfig",
    "PeftConfig",
    "SFTParams",
    "TrainingConfig",
]

AlignmentMethod = Literal["sft", "dpo", "grpo"]
"""The alignment algorithms AlignPy currently supports."""

DPOLossType = Literal[
    "sigmoid",  # classic DPO (Rafailov et al., 2023)
    "hinge",    # SLiC-HF hinge loss
    "ipo",      # IPO (Azar et al., 2023) — beta is the tau regularizer
    "robust",   # label-noise-robust DPO
    "exo_pair",
    "nca_pair",
    "sppo_hard",
    "apo_zero",
    "apo_down",
]
"""Preference losses forwarded verbatim to ``trl.DPOConfig.loss_type``."""


class _StrictModel(BaseModel):
    """Base for all AlignPy config blocks: unknown keys are hard errors."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=())


class ModelConfig(_StrictModel):
    """Which model(s) to align and how to load them."""

    model_name_or_path: str = Field(
        description="Hub id or local path of the (post-SFT) policy model to align."
    )
    ref_model_name_or_path: str | None = Field(
        default=None,
        description=(
            "Explicit reference model for DPO/GRPO KL terms. Leave unset to let TRL "
            "derive one (a frozen copy of the policy, or disabled adapters under PEFT)."
        ),
    )
    torch_dtype: Literal["auto", "bfloat16", "float16", "float32"] = Field(
        default="auto", description="Dtype passed to ``from_pretrained``."
    )
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] | None = Field(
        default=None, description="Attention backend; None uses the transformers default."
    )
    trust_remote_code: bool = Field(
        default=False, description="Allow custom modeling code from the Hub."
    )


class PeftConfig(_StrictModel):
    """Optional LoRA adaptation (via ``peft``) for memory-efficient alignment."""

    enabled: bool = Field(default=False, description="Train LoRA adapters instead of full weights.")
    r: int = Field(default=16, ge=1, description="LoRA rank.")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA scaling factor.")
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0, description="LoRA dropout.")
    target_modules: list[str] | Literal["all-linear"] = Field(
        default="all-linear", description="Module names to adapt, or 'all-linear'."
    )


class DataConfig(_StrictModel):
    """Dataset location and column mapping.

    Column fields map *your* dataset's schema onto the canonical column names
    each TRL trainer expects (``prompt``/``chosen``/``rejected`` for DPO,
    ``prompt`` for GRPO, ``text`` or ``messages`` for SFT). Columns already
    canonically named need no mapping.
    """

    dataset_name_or_path: str | None = Field(
        default=None,
        description=(
            "Hub dataset id or local path (anything ``datasets.load_dataset`` accepts). "
            "Optional when datasets are passed to the trainer directly (SDK usage)."
        ),
    )
    split: str = Field(default="train", description="Training split expression, e.g. 'train[:5%]'.")
    eval_split: str | None = Field(
        default=None, description="Optional held-out split for in-training evaluation."
    )
    prompt_column: str = Field(default="prompt")
    chosen_column: str = Field(default="chosen", description="Preferred response (DPO).")
    rejected_column: str = Field(default="rejected", description="Dispreferred response (DPO).")
    text_column: str = Field(default="text", description="Full formatted sample (SFT).")
    max_samples: int | None = Field(
        default=None, ge=1, description="Optional cap on samples — handy for smoke tests."
    )


class TrainingConfig(_StrictModel):
    """Optimizer, schedule, precision, and logging — shared by every method."""

    output_dir: str = Field(default="./alignpy-output", description="Checkpoint/output directory.")
    learning_rate: float = Field(default=1e-6, gt=0.0)
    per_device_train_batch_size: int = Field(default=4, ge=1)
    per_device_eval_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    num_train_epochs: float = Field(default=1.0, gt=0.0)
    max_steps: int = Field(
        default=-1, description="If > 0, overrides num_train_epochs (transformers semantics)."
    )
    lr_scheduler_type: str = Field(default="cosine")
    warmup_steps: int = Field(default=0, ge=0)
    logging_steps: int = Field(default=10, ge=1)
    save_strategy: Literal["no", "steps", "epoch"] = Field(default="epoch")
    save_steps: int = Field(default=500, ge=1)
    eval_strategy: Literal["no", "steps", "epoch"] = Field(
        default="no", description="Set to 'steps'/'epoch' when data.eval_split is provided."
    )
    eval_steps: int = Field(default=500, ge=1)
    bf16: bool = Field(default=False)
    fp16: bool = Field(default=False)
    gradient_checkpointing: bool = Field(default=False)
    seed: int = Field(default=42)
    report_to: str | list[str] = Field(
        default="none", description="Experiment trackers, e.g. 'wandb' (transformers semantics)."
    )

    @model_validator(mode="after")
    def _check_precision(self) -> TrainingConfig:
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 are mutually exclusive; enable at most one.")
        return self


class SFTParams(_StrictModel):
    """Hyperparameters for supervised fine-tuning (instruction baselines)."""

    max_length: int = Field(
        default=2048, ge=8, description="Max tokenized sample length (truncated beyond)."
    )
    packing: bool = Field(
        default=False, description="Pack multiple short samples per sequence for throughput."
    )


class DPOParams(_StrictModel):
    """Hyperparameters for Direct Preference Optimization.

    The implicit reward of a response is ``beta * log(pi(y|x) / pi_ref(y|x))``;
    training pushes the chosen/rejected *margin* of that quantity apart. TRL
    logs it as ``rewards/margins``, which AlignPy surfaces through reward hooks.
    """

    beta: float = Field(
        default=0.1, gt=0.0,
        description="KL-anchoring strength; higher stays closer to the reference model.",
    )
    loss_type: DPOLossType = Field(default="sigmoid")
    label_smoothing: float = Field(
        default=0.0, ge=0.0, lt=0.5,
        description="Assumed preference-label noise (used by 'sigmoid'/'robust' losses).",
    )
    max_prompt_length: int = Field(default=512, ge=8)
    max_completion_length: int = Field(default=1024, ge=8)
    max_length: int | None = Field(
        default=None,
        description=(
            "Full-sequence cap; defaults to max_prompt_length + max_completion_length. "
            "This combined cap is what recent TRL (>= 1.0) enforces at tokenization time."
        ),
    )
    truncation_mode: Literal["keep_start", "keep_end"] = Field(
        default="keep_start", description="Which side of over-long sequences to keep."
    )

    @model_validator(mode="after")
    def _derive_max_length(self) -> DPOParams:
        if self.max_length is None:
            # Assign through __dict__ to avoid re-triggering validate_assignment.
            self.__dict__["max_length"] = self.max_prompt_length + self.max_completion_length
        elif self.max_length < self.max_prompt_length:
            raise ValueError(
                f"max_length ({self.max_length}) must be >= max_prompt_length "
                f"({self.max_prompt_length})."
            )
        return self


class GRPOParams(_StrictModel):
    """Hyperparameters for Group Relative Policy Optimization.

    GRPO samples ``num_generations`` completions per prompt, scores each with
    the reward pipeline, and normalizes rewards *within the group* to form
    advantages — no value network required. Rewards come from
    :mod:`alignpy.rewards`: names listed in ``reward_funcs`` are resolved from
    the registry, and extra callables can be passed to the trainer directly.
    """

    num_generations: int = Field(
        default=8, ge=2, description="Group size G: completions sampled per prompt."
    )
    max_completion_length: int = Field(default=1024, ge=8)
    temperature: float = Field(default=0.9, gt=0.0, description="Sampling temperature.")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    beta: float = Field(
        default=0.04, ge=0.0, description="KL penalty toward the reference model (0 disables)."
    )
    epsilon: float = Field(default=0.2, gt=0.0, description="PPO-style clipping range.")
    reward_funcs: list[str] = Field(
        default_factory=list,
        description="Registry names of reward functions (see alignpy.rewards.register_reward).",
    )
    reward_weights: list[float] | None = Field(
        default=None, description="Per-function weights; defaults to 1.0 each."
    )
    scale_rewards: Literal["group", "batch", "none"] = Field(
        default="group",
        description="Reward-std normalization scope for advantages (TRL >= 1.0 semantics).",
    )

    @model_validator(mode="after")
    def _check_weights(self) -> GRPOParams:
        if self.reward_weights is not None and len(self.reward_weights) != len(self.reward_funcs):
            raise ValueError(
                f"reward_weights has {len(self.reward_weights)} entries but reward_funcs "
                f"has {len(self.reward_funcs)}; they must match."
            )
        return self


class AlignmentConfig(_StrictModel):
    """Root configuration for one alignment run.

    Example
    -------
    >>> config = AlignmentConfig.from_dict({
    ...     "method": "dpo",
    ...     "model": {"model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"},
    ...     "data": {"dataset_name_or_path": "trl-lib/ultrafeedback_binarized"},
    ...     "dpo": {"beta": 0.1, "loss_type": "sigmoid"},
    ... })
    >>> config.dpo.max_length  # derived from prompt + completion lengths
    1536
    """

    method: AlignmentMethod = Field(description="Which alignment algorithm to run.")
    model: ModelConfig
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Optional when datasets are passed to the trainer directly.",
    )
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    peft: PeftConfig = Field(default_factory=PeftConfig)
    sft: SFTParams | None = None
    dpo: DPOParams | None = None
    grpo: GRPOParams | None = None

    @model_validator(mode="after")
    def _ensure_active_params(self) -> AlignmentConfig:
        """Auto-create the parameter block for the selected method if omitted."""
        defaults = {"sft": SFTParams, "dpo": DPOParams, "grpo": GRPOParams}
        if getattr(self, self.method) is None:
            self.__dict__[self.method] = defaults[self.method]()
        return self

    @property
    def active_params(self) -> SFTParams | DPOParams | GRPOParams:
        """The parameter block matching ``method`` (guaranteed non-None)."""
        params = getattr(self, self.method)
        assert params is not None  # enforced by _ensure_active_params
        return params

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentConfig:
        """Validate a plain dict (e.g. parsed YAML/JSON) into a config."""
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AlignmentConfig:
        """Load and validate a YAML config file."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}.")
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> Path:
        """Serialize this config to YAML (exact round-trip via ``from_yaml``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
        return path

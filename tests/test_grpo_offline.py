"""Hub-free GRPO integration test: real generation + multi-reward pipeline.

Exercises the full reward pipeline in one optimizer step: a registry-named
reward function (``length_penalty``) from the config, an ad-hoc callable
passed to the trainer (which must get a default weight of 1.0 appended to the
config-declared weights), group sampling with num_generations=2, and the
reward-hook telemetry (``reward`` / ``rewards/<name>/...`` keys).
"""

import math

import pytest

pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("trl", reason="requires trl")
pytest.importorskip("datasets", reason="requires datasets")

from datasets import Dataset

from alignpy import AlignmentConfig, GRPOTrainer


def prefer_short(completions, **kwargs):
    """Ad-hoc reward: higher for shorter completions."""
    return [-float(len(str(completion))) / 100.0 for completion in completions]


def test_grpo_one_step_with_mixed_rewards(tiny_model_dir, tmp_path):
    dataset = Dataset.from_dict({
        "prompt": [
            "What is the capital of France?",
            "How many legs does a spider have?",
        ]
    })

    config = AlignmentConfig.from_dict({
        "method": "grpo",
        "model": {"model_name_or_path": str(tiny_model_dir), "torch_dtype": "float32"},
        "train": {
            "output_dir": str(tmp_path / "tiny-grpo"),
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 2,
            "max_steps": 1,
            "logging_steps": 1,
            "save_strategy": "no",
            "seed": 42,
        },
        "grpo": {
            "num_generations": 2,
            "max_completion_length": 8,
            "temperature": 1.0,
            "beta": 0.0,
            "reward_funcs": ["length_penalty"],
            "reward_weights": [0.5],
        },
    })

    reward_log: list[dict[str, float]] = []
    trainer = GRPOTrainer(
        config,
        train_dataset=dataset,
        reward_funcs=[prefer_short],
        reward_hooks=[lambda metrics, step: reward_log.append(dict(metrics))],
    )
    result = trainer.train()

    assert result.global_step == 1
    assert math.isfinite(result.training_loss)

    # The hook saw reward telemetry, and every reported value is finite.
    assert reward_log, "reward hooks never fired"
    merged = {key: value for entry in reward_log for key, value in entry.items()}
    assert "reward" in merged
    assert all(math.isfinite(value) for value in merged.values())
    # Both reward sources show up in TRL's per-function telemetry.
    per_func_keys = [key for key in merged if key.startswith("rewards/")]
    assert any("length_penalty" in key for key in per_func_keys)
    assert any("prefer_short" in key for key in per_func_keys)


def test_grpo_requires_at_least_one_reward(tiny_model_dir, tmp_path):
    config = AlignmentConfig.from_dict({
        "method": "grpo",
        "model": {"model_name_or_path": str(tiny_model_dir)},
        "train": {"output_dir": str(tmp_path / "x")},
        "grpo": {"reward_funcs": []},
    })
    trainer = GRPOTrainer(config, train_dataset=Dataset.from_dict({"prompt": ["hi"]}))
    with pytest.raises(ValueError, match="at least one reward"):
        trainer.train()

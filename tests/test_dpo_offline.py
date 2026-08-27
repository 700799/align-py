"""Hub-free DPO integration test: tiny random Llama + LoRA, 2 steps, offline.

Runs the *identical* AlignPy -> TRL code path as ``test_dpo_smollm2.py`` —
config validation, in-memory preference dataset, LoRA attachment, two real
optimizer steps, reward-hook telemetry — but against the tiny local model from
``conftest.tiny_model_dir``, so it needs no network access and finishes in
seconds. The correctness invariants asserted here (zero-init LoRA => policy ==
reference at step 1 => implicit reward margin ~0 and sigmoid DPO loss ~ln 2)
are architecture-independent.
"""

import math

import pytest

pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("trl", reason="requires trl")
pytest.importorskip("peft", reason="requires peft")
pytest.importorskip("datasets", reason="requires datasets")

from datasets import Dataset

from alignpy import AlignmentConfig, DPOTrainer
from tests.conftest import PREFERENCE_PAIRS


def test_dpo_two_steps_offline_tiny_model(tiny_model_dir, tmp_path):
    dataset = Dataset.from_dict(PREFERENCE_PAIRS)

    config = AlignmentConfig.from_dict({
        "method": "dpo",
        "model": {"model_name_or_path": str(tiny_model_dir), "torch_dtype": "float32"},
        "peft": {"enabled": True, "r": 8, "lora_alpha": 16,
                 "target_modules": ["q_proj", "v_proj"]},
        "train": {
            "output_dir": str(tmp_path / "tiny-dpo"),
            "learning_rate": 5e-4,
            "per_device_train_batch_size": 3,
            "max_steps": 2,
            "logging_steps": 1,
            "save_strategy": "no",
            "seed": 42,
        },
        "dpo": {"beta": 0.1, "loss_type": "sigmoid",
                "max_prompt_length": 64, "max_completion_length": 128},
    })

    margin_log: list[tuple[int, float]] = []

    def capture_margins(metrics, step):
        if "rewards/margins" in metrics:
            margin_log.append((step, metrics["rewards/margins"]))

    trainer = DPOTrainer(config, train_dataset=dataset, reward_hooks=[capture_margins])
    result = trainer.train()

    print("\nImplicit log-ratio reward margins (tiny offline model):")
    for step, margin in margin_log:
        print(f"  step {step}: rewards/margins = {margin:+.6f}")

    assert result.global_step == 2
    assert math.isfinite(result.training_loss)
    assert [step for step, _ in margin_log] == [1, 2]
    assert all(math.isfinite(margin) for _, margin in margin_log)

    # Zero-init LoRA: policy == reference at step 1 => margin ~0, loss ~ln 2.
    assert abs(margin_log[0][1]) < 1e-4
    assert trainer.trainer.state.log_history[0]["loss"] == pytest.approx(math.log(2), abs=0.05)

    # One optimizer step moved the policy off the reference.
    assert margin_log[1][1] != pytest.approx(margin_log[0][1], abs=1e-9)

    # Only LoRA adapters are trainable.
    model = trainer.trainer.model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < trainable < 0.1 * total

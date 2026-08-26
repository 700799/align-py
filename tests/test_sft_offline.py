"""Hub-free SFT integration test: 2 real optimizer steps + save round-trip."""

import math

import pytest

pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("trl", reason="requires trl")
pytest.importorskip("datasets", reason="requires datasets")

from datasets import Dataset

from alignpy import AlignmentConfig, SFTTrainer
from alignpy.trainers.base import CONFIG_SNAPSHOT_FILENAME


def test_sft_two_steps_and_save_roundtrip(tiny_model_dir, tmp_path):
    # Custom column name ("content") exercises the DataConfig column mapping.
    dataset = Dataset.from_dict({
        "content": [
            "The capital of France is Paris.",
            "A spider has eight legs.",
            "The sky is blue on a clear day.",
            "Water freezes at zero degrees Celsius.",
        ]
    })

    output_dir = tmp_path / "tiny-sft"
    config = AlignmentConfig.from_dict({
        "method": "sft",
        "model": {"model_name_or_path": str(tiny_model_dir), "torch_dtype": "float32"},
        "data": {"text_column": "content"},
        "train": {
            "output_dir": str(output_dir),
            "learning_rate": 5e-4,
            "per_device_train_batch_size": 2,
            "max_steps": 2,
            "logging_steps": 1,
            "save_strategy": "no",
            "seed": 42,
        },
        "sft": {"max_length": 64, "packing": False},
    })

    trainer = SFTTrainer(config, train_dataset=dataset)
    result = trainer.train()

    assert result.global_step == 2
    assert math.isfinite(result.training_loss)
    assert result.training_loss > 0

    # save() writes model + tokenizer + a config snapshot that round-trips.
    saved_dir = trainer.save()
    assert saved_dir == output_dir
    assert (saved_dir / CONFIG_SNAPSHOT_FILENAME).is_file()
    assert any(saved_dir.glob("*.safetensors")), "model weights not saved"
    assert AlignmentConfig.from_yaml(saved_dir / CONFIG_SNAPSHOT_FILENAME) == config

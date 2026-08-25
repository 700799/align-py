"""Hub-free DPO integration test: tiny random Llama + LoRA, 2 steps, offline.

Runs the *identical* AlignPy -> TRL code path as ``test_dpo_smollm2.py`` —
config validation, in-memory preference dataset, LoRA attachment, two real
optimizer steps, reward-hook telemetry — but against a ~400k-parameter
randomly-initialized Llama-architecture model built and saved locally, so it
needs no network access and finishes in seconds. The correctness invariants
asserted here (zero-init LoRA => policy == reference at step 1 => implicit
reward margin ~0 and sigmoid DPO loss ~ln 2) are architecture-independent.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("trl", reason="requires trl")
pytest.importorskip("peft", reason="requires peft")
pytest.importorskip("datasets", reason="requires datasets")

from datasets import Dataset

from alignpy import AlignmentConfig, DPOTrainer

PREFERENCE_PAIRS = {
    "prompt": [
        "What is the capital of France?",
        "How many legs does a spider have?",
        "What color is the sky on a clear day?",
    ],
    "chosen": [
        "The capital of France is Paris.",
        "A spider has eight legs.",
        "The sky is blue on a clear day.",
    ],
    "rejected": [
        "Well, that is a really interesting question, and after this long preamble the "
        "answer is of course the world-famous city of Paris.",
        "Great question! Spiders are fascinating arachnids famously known throughout the "
        "animal kingdom for possessing a grand total of eight legs altogether.",
        "Ah, the sky! To human observers it appears to be a beautiful shade of blue due "
        "to a phenomenon called Rayleigh scattering.",
    ],
}


def _build_tiny_llama(save_dir) -> None:
    """Save a tiny random Llama model + freshly-trained BPE tokenizer locally."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    corpus = sum(PREFERENCE_PAIRS.values(), [])
    bpe = Tokenizer(models.BPE(unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.Whitespace()
    bpe.train_from_iterator(
        corpus, trainers.BpeTrainer(vocab_size=512, special_tokens=["<unk>", "<s>", "</s>"])
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
    )
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def test_dpo_two_steps_offline_tiny_model(tmp_path):
    model_dir = tmp_path / "tiny-llama"
    _build_tiny_llama(model_dir)
    dataset = Dataset.from_dict(PREFERENCE_PAIRS)

    config = AlignmentConfig.from_dict({
        "method": "dpo",
        "model": {"model_name_or_path": str(model_dir), "torch_dtype": "float32"},
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
    trainer = DPOTrainer(
        config,
        train_dataset=dataset,
        reward_hooks=[lambda m, step: margin_log.append((step, m["rewards/margins"]))
                      if "rewards/margins" in m else None],
    )
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

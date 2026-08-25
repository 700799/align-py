"""End-to-end DPO alignment smoke test: SmolLM2-135M-Instruct + LoRA, 2 steps.

Exercises the full AlignPy stack against real TRL machinery — config
validation, in-memory dataset injection, LoRA attachment via ``peft``, TRL's
``DPOTrainer`` under AlignPy's wrapper, and the reward-hook pipeline that
surfaces implicit log-ratio reward margins (``beta * [log pi/pi_ref](chosen) -
beta * [log pi/pi_ref](rejected)``).

Two math facts make this a *correctness* test rather than just a crash test:

1. LoRA's B matrices initialize to zero, so at step 1 the policy is exactly
   the reference model. The implicit reward margin must therefore start at ~0
   and the sigmoid DPO loss at ``-log(1/2) = ln 2 ~= 0.6931``.
2. After one gradient step the policy moves, so step 2's telemetry must differ.

Runtime: ~1 min CPU after a one-time ~270MB model download. Marked
``integration`` — deselect with ``pytest -m "not integration"`` for the fast lane.
"""

import math

import pytest

torch = pytest.importorskip("torch", reason="integration test requires torch")
pytest.importorskip("trl", reason="integration test requires trl")
pytest.importorskip("peft", reason="integration test requires peft")
pytest.importorskip("datasets", reason="integration test requires datasets")

from datasets import Dataset

from alignpy import AlignmentConfig, DPOTrainer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _hub_reachable() -> bool:
    """True if the Hugging Face Hub is reachable (needed to download the model)."""
    import os

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    import urllib.request

    try:
        urllib.request.urlopen("https://huggingface.co", timeout=10)
        return True
    except Exception:
        return False

# Synthetic preference data: concise (chosen) vs. wordy (rejected) answers.
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
        "Well, that's a really interesting question about European geography! France is a "
        "country with a long and storied history, and its capital city, which I will now "
        "reveal after this preamble, is of course the world-famous city of Paris.",
        "Great question! Spiders are fascinating arachnids, and unlike insects, which have "
        "six legs, spiders are famously known throughout the animal kingdom for possessing "
        "a grand total of eight legs altogether.",
        "Ah, the sky! On a clear day, when there are no clouds and the sun is shining "
        "brightly, the sky appears to human observers to be a beautiful shade of blue due "
        "to a phenomenon called Rayleigh scattering.",
    ],
}


@pytest.mark.integration
def test_dpo_two_steps_smollm2(tmp_path):
    if not _hub_reachable():
        pytest.skip("Hugging Face Hub unreachable; cannot download SmolLM2-135M-Instruct "
                    "(see tests/test_dpo_offline.py for the hub-free equivalent)")
    dataset = Dataset.from_dict(PREFERENCE_PAIRS)

    config = AlignmentConfig.from_dict({
        "method": "dpo",
        "model": {"model_name_or_path": MODEL_ID, "torch_dtype": "float32"},
        "peft": {"enabled": True, "r": 8, "lora_alpha": 16,
                 "target_modules": ["q_proj", "v_proj"]},
        "train": {
            "output_dir": str(tmp_path / "smollm2-dpo"),
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 3,   # all 3 pairs per optimizer step
            "max_steps": 2,
            "logging_steps": 1,                 # reward telemetry at every step
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

    print("\nImplicit log-ratio reward margins (beta * Δ log[pi/pi_ref]):")
    for step, margin in margin_log:
        print(f"  step {step}: rewards/margins = {margin:+.6f}")

    # Ran exactly the requested 2 optimizer steps.
    assert result.global_step == 2
    assert math.isfinite(result.training_loss)

    # The reward-hook pipeline fired once per logging step with finite margins.
    assert [step for step, _ in margin_log] == [1, 2]
    assert all(math.isfinite(margin) for _, margin in margin_log)

    # Step 1: zero-initialized LoRA means policy == reference, so the implicit
    # reward margin is ~0 and the sigmoid DPO loss is ~ln 2.
    first_loss = trainer.trainer.state.log_history[0]["loss"]
    assert abs(margin_log[0][1]) < 1e-4
    assert first_loss == pytest.approx(math.log(2), abs=0.05)

    # The optimizer actually moved the policy off the reference between steps.
    assert margin_log[1][1] != pytest.approx(margin_log[0][1], abs=1e-9)

    # LoRA kept the run parameter-efficient: only adapters are trainable.
    model = trainer.trainer.model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < trainable < 0.02 * total

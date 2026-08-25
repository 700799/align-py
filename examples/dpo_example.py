"""Minimal AlignPy DPO run: align a post-SFT model on preference pairs."""
from alignpy import AlignmentConfig, DPOTrainer

config = AlignmentConfig.from_dict({
    "method": "dpo",
    "model": {"model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "torch_dtype": "bfloat16"},
    "data": {"dataset_name_or_path": "trl-lib/ultrafeedback_binarized", "split": "train[:1%]"},
    "train": {"output_dir": "./qwen-dpo", "learning_rate": 5e-6, "bf16": True},
    "dpo": {"beta": 0.1, "loss_type": "sigmoid", "max_prompt_length": 512},
    "peft": {"enabled": True, "r": 16},
})

def watch_margins(metrics, step):  # fires every logging step with TRL's reward telemetry
    print(f"[step {step}] implicit reward margin = {metrics.get('rewards/margins', 0.0):+.4f}")

trainer = DPOTrainer(config, reward_hooks=[watch_margins])
trainer.train()
trainer.save()  # writes model + tokenizer + alignpy_config.yaml -> `alignpy eval --model ./qwen-dpo`

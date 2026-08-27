"""Unit tests for trainer plumbing: hooks, dispatch, guards, dataset prep.

These need torch/trl importable but never load a model — construction is lazy
by design, so everything here runs in milliseconds.
"""

import pytest

pytest.importorskip("torch", reason="requires torch")
pytest.importorskip("trl", reason="requires trl")
pytest.importorskip("datasets", reason="requires datasets")

from datasets import Dataset
from transformers import TrainerState

from alignpy import AlignmentConfig
from alignpy.trainers import DPOTrainer, GRPOTrainer, SFTTrainer, build_trainer
from alignpy.trainers.base import RewardMetricsCallback


def make_config(method="dpo", **overrides):
    base = {
        "method": method,
        "model": {"model_name_or_path": "dummy/never-loaded"},
        "train": {"output_dir": "./never-created"},
    }
    base.update(overrides)
    return AlignmentConfig.from_dict(base)


# --------------------------------------------------------------------------- #
# RewardMetricsCallback
# --------------------------------------------------------------------------- #

def fire(callback, logs, step=7):
    state = TrainerState()
    state.global_step = step
    callback.on_log(args=None, state=state, control=None, logs=logs)


def test_callback_filters_reward_keys_only():
    seen = []
    fire(
        RewardMetricsCallback([lambda metrics, step: seen.append((metrics, step))]),
        {"loss": 0.69, "rewards/margins": 0.5, "eval_rewards/margins": 0.4,
         "reward": 1.0, "reward_std": 0.1, "learning_rate": 1e-5},
    )
    assert seen == [({"rewards/margins": 0.5, "eval_rewards/margins": 0.4,
                      "reward": 1.0, "reward_std": 0.1}, 7)]


def test_callback_silent_without_reward_keys_or_logs():
    seen = []
    callback = RewardMetricsCallback([lambda metrics, step: seen.append(metrics)])
    fire(callback, {"loss": 0.69, "epoch": 1.0})
    fire(callback, None)
    fire(callback, {})
    assert seen == []


def test_callback_ignores_non_numeric_values_and_fans_out():
    first, second = [], []
    fire(
        RewardMetricsCallback([lambda m, s: first.append(m), lambda m, s: second.append(m)]),
        {"rewards/margins": 0.25, "rewards/label": "not-a-number"},
    )
    assert first == second == [{"rewards/margins": 0.25}]


# --------------------------------------------------------------------------- #
# Dispatch and guards
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "method,cls", [("sft", SFTTrainer), ("dpo", DPOTrainer), ("grpo", GRPOTrainer)]
)
def test_build_trainer_dispatch(method, cls):
    assert isinstance(build_trainer(make_config(method)), cls)


def test_method_mismatch_raises():
    with pytest.raises(ValueError, match="method='dpo'.*method='sft'"):
        DPOTrainer(make_config("sft"))


def test_evaluate_without_eval_data_raises_before_any_loading():
    trainer = DPOTrainer(make_config(), train_dataset=Dataset.from_dict({"prompt": ["x"]}))
    with pytest.raises(ValueError, match="No eval data"):
        trainer.evaluate()


def test_missing_train_data_raises_clearly():
    trainer = DPOTrainer(make_config())  # no dataset injected, no dataset in config
    with pytest.raises(ValueError, match="No training data"):
        trainer._load_datasets(column_mapping={})


def test_eval_strategy_without_eval_dataset_raises():
    config = make_config(train={"output_dir": "./x", "eval_strategy": "steps"})
    trainer = DPOTrainer(config, train_dataset=Dataset.from_dict({"prompt": ["x"]}))
    with pytest.raises(ValueError, match="eval_strategy"):
        trainer._load_datasets(column_mapping={})


# --------------------------------------------------------------------------- #
# Dataset preparation
# --------------------------------------------------------------------------- #

def test_load_datasets_renames_columns_and_caps_samples():
    config = make_config(
        data={"prompt_column": "question", "chosen_column": "good",
              "rejected_column": "bad", "max_samples": 2},
    )
    raw = Dataset.from_dict({
        "question": ["q1", "q2", "q3"],
        "good": ["g1", "g2", "g3"],
        "bad": ["b1", "b2", "b3"],
    })
    trainer = DPOTrainer(config, train_dataset=raw)
    train, eval_ds = trainer._load_datasets(
        column_mapping={"question": "prompt", "good": "chosen", "bad": "rejected"}
    )
    assert eval_ds is None
    assert sorted(train.column_names) == ["chosen", "prompt", "rejected"]
    assert len(train) == 2
    assert train[0] == {"prompt": "q1", "chosen": "g1", "rejected": "b1"}


def test_load_datasets_leaves_canonical_columns_alone():
    config = make_config()
    raw = Dataset.from_dict({"prompt": ["q"], "chosen": ["g"], "rejected": ["b"]})
    trainer = DPOTrainer(config, train_dataset=raw)
    train, _ = trainer._load_datasets(
        column_mapping={"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}
    )
    assert sorted(train.column_names) == ["chosen", "prompt", "rejected"]
    assert len(train) == 1

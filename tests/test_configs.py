"""Schema tests for AlignmentConfig — runnable without torch/trl installed."""

import pytest
from pydantic import ValidationError

from alignpy.configs import AlignmentConfig, DPOParams, GRPOParams, SFTParams

MINIMAL = {
    "method": "dpo",
    "model": {"model_name_or_path": "org/model"},
    "data": {"dataset_name_or_path": "org/dataset"},
}


def test_minimal_config_fills_defaults():
    config = AlignmentConfig.from_dict(MINIMAL)
    assert config.method == "dpo"
    assert config.dpo is not None, "active params block should be auto-created"
    assert config.dpo.beta == 0.1
    assert config.train.output_dir == "./alignpy-output"
    assert config.active_params is config.dpo


@pytest.mark.parametrize(
    "method,params_cls",
    [("sft", SFTParams), ("dpo", DPOParams), ("grpo", GRPOParams)],
)
def test_active_params_per_method(method, params_cls):
    config = AlignmentConfig.from_dict({**MINIMAL, "method": method})
    assert isinstance(config.active_params, params_cls)


def test_unknown_key_rejected():
    with pytest.raises(ValidationError, match="betta"):
        AlignmentConfig.from_dict({**MINIMAL, "dpo": {"betta": 0.1}})


def test_negative_beta_rejected():
    with pytest.raises(ValidationError):
        AlignmentConfig.from_dict({**MINIMAL, "dpo": {"beta": -0.1}})


def test_bad_loss_type_rejected():
    with pytest.raises(ValidationError):
        AlignmentConfig.from_dict({**MINIMAL, "dpo": {"loss_type": "not_a_loss"}})


def test_dpo_max_length_derived():
    config = AlignmentConfig.from_dict(
        {**MINIMAL, "dpo": {"max_prompt_length": 100, "max_completion_length": 200}}
    )
    assert config.dpo.max_length == 300


def test_dpo_max_length_explicit_too_small():
    with pytest.raises(ValidationError, match="max_length"):
        DPOParams(max_prompt_length=512, max_length=100)


def test_bf16_fp16_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        AlignmentConfig.from_dict({**MINIMAL, "train": {"bf16": True, "fp16": True}})


def test_grpo_reward_weights_must_match_funcs():
    with pytest.raises(ValidationError, match="reward_weights"):
        GRPOParams(reward_funcs=["a", "b"], reward_weights=[1.0])


def test_grpo_num_generations_minimum():
    with pytest.raises(ValidationError):
        GRPOParams(num_generations=1)


def test_yaml_round_trip(tmp_path):
    config = AlignmentConfig.from_dict(
        {**MINIMAL, "dpo": {"beta": 0.25, "loss_type": "ipo"}, "peft": {"enabled": True, "r": 8}}
    )
    path = config.to_yaml(tmp_path / "config.yaml")
    assert AlignmentConfig.from_yaml(path) == config


def test_data_block_optional_for_sdk_usage():
    config = AlignmentConfig.from_dict({"method": "dpo", "model": {"model_name_or_path": "m"}})
    assert config.data.dataset_name_or_path is None
    assert config.data.split == "train"


def test_truncation_mode_and_scale_rewards_literals_enforced():
    with pytest.raises(ValidationError):
        DPOParams(truncation_mode="keep_middle")
    with pytest.raises(ValidationError):
        GRPOParams(scale_rewards="sometimes")
    assert GRPOParams(scale_rewards="none").scale_rewards == "none"


def test_negative_warmup_steps_rejected():
    with pytest.raises(ValidationError):
        AlignmentConfig.from_dict({**MINIMAL, "train": {"warmup_steps": -1}})


def test_example_configs_validate():
    from pathlib import Path

    examples = Path(__file__).parent.parent / "examples" / "configs"
    for yaml_file in sorted(examples.glob("*.yaml")):
        config = AlignmentConfig.from_yaml(yaml_file)
        assert config.active_params is not None

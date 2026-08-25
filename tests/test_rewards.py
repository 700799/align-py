"""Tests for the reward registry and built-in reward functions."""

import pytest

from alignpy.rewards import (
    get_reward,
    length_penalty,
    register_reward,
    registered_rewards,
    resolve_reward,
)


def test_builtin_is_registered():
    assert get_reward("length_penalty") is length_penalty


def test_register_and_resolve():
    @register_reward("test_always_one")
    def always_one(completions, **kwargs):
        return [1.0] * len(completions)

    assert resolve_reward("test_always_one") is always_one
    assert "test_always_one" in registered_rewards()


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_reward("length_penalty")(lambda completions, **kw: [])


def test_unregistered_name_passes_through_as_model_id():
    assert resolve_reward("org/some-reward-model") == "org/some-reward-model"


def test_get_reward_unknown_raises_with_hints():
    with pytest.raises(KeyError, match="length_penalty"):
        get_reward("nope")


def test_length_penalty_scoring():
    short = "ok"
    at_target = "x" * 1024
    double = "x" * 2048
    scores = length_penalty([short, at_target, double])
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == pytest.approx(-1.0)


def test_length_penalty_chat_format():
    conversation = [{"role": "assistant", "content": "x" * 3000}]
    (score,) = length_penalty([conversation])
    assert score == -1.0

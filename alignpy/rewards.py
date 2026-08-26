"""Dynamic reward pipeline for AlignPy.

GRPO (and any future reward-based trainer) scores sampled completions with one
or more *reward functions*. AlignPy supports two kinds side by side:

1. **Custom Python functions** — anything matching :class:`RewardFunction`
   (length penalties, code-execution success, tone checks, ...). Register them
   by name so YAML configs can reference them, or pass callables straight to
   the trainer.
2. **Reward models** — a Hub model id in ``GRPOParams.reward_funcs`` that is
   *not* in the registry is forwarded verbatim to TRL, which loads it as a
   ``AutoModelForSequenceClassification`` reward model.

Example
-------
>>> from alignpy.rewards import register_reward
>>>
>>> @register_reward("ends_politely")
... def ends_politely(completions, **kwargs):
...     return [1.0 if c.rstrip().endswith(("!", ".")) else 0.0 for c in completions]

then in YAML::

    grpo:
      reward_funcs: [ends_politely, length_penalty]
      reward_weights: [1.0, 0.2]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "RewardFunction",
    "get_reward",
    "length_penalty",
    "register_reward",
    "registered_rewards",
    "resolve_reward",
]


@runtime_checkable
class RewardFunction(Protocol):
    """A batched reward function, call-compatible with TRL's GRPO trainer.

    Receives the batch of sampled ``completions`` plus keyword arguments for
    every other dataset column (including ``prompts``), and returns one float
    score per completion. Higher is better; scale is up to you — GRPO
    normalizes rewards within each generation group anyway.
    """

    __name__: str

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]:
        ...


_REGISTRY: dict[str, RewardFunction] = {}


def register_reward(
    name: str | None = None,
) -> Callable[[RewardFunction], RewardFunction]:
    """Decorator: register a reward function under ``name`` (default: its ``__name__``).

    Registered names can be referenced from ``GRPOParams.reward_funcs`` in YAML
    configs. Re-registering an existing name raises to prevent silent shadowing.
    """

    def decorator(func: RewardFunction) -> RewardFunction:
        key = name or func.__name__
        if key in _REGISTRY:
            raise ValueError(f"A reward function named {key!r} is already registered.")
        _REGISTRY[key] = func
        return func

    return decorator


def get_reward(name: str) -> RewardFunction:
    """Look up a registered reward function by name (KeyError with hints if absent)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"No reward function registered as {name!r}. Registered: {known}."
        ) from None


def registered_rewards() -> dict[str, RewardFunction]:
    """A snapshot of the current registry (name -> function)."""
    return dict(_REGISTRY)


def resolve_reward(spec: str) -> RewardFunction | str:
    """Resolve a config entry: a registered function, else assume a reward-model id.

    Strings that are not registered names are returned unchanged so TRL can
    load them as sequence-classification reward models.
    """
    return _REGISTRY.get(spec, spec)


# --------------------------------------------------------------------------- #
# Built-ins — small, dependency-free examples users can reference immediately.
# --------------------------------------------------------------------------- #

@register_reward("length_penalty")
def length_penalty(
    completions: list[Any], *, target_chars: int = 1024, **kwargs: Any
) -> list[float]:
    """Penalize completions that overrun ``target_chars`` characters.

    Returns 0.0 at or under the target, scaling linearly to -1.0 at twice the
    target (clamped below). Chat-style completions (lists of message dicts) are
    scored on their concatenated text content.
    """
    def text_of(completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        # Conversational format: [{"role": ..., "content": ...}, ...]
        return "".join(str(m.get("content", "")) for m in completion)

    scores = []
    for completion in completions:
        overrun = len(text_of(completion)) - target_chars
        scores.append(0.0 if overrun <= 0 else max(-1.0, -overrun / target_chars))
    return scores

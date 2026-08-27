# AlignPy

A lightweight, highly modular Python framework for LLM alignment, preference
tuning, and reasoning optimization — built on Hugging Face
[TRL](https://github.com/huggingface/trl), `peft`, and `transformers`.

AlignPy takes a post-SFT base model and aligns it using human preferences or
automated rewards, through a clean Python-first SDK and a YAML-driven CLI:

- **SFT** — supervised fine-tuning for instruction-following baselines.
- **DPO** — Direct Preference Optimization over chosen/rejected pairs, with
  live implicit-reward-margin tracking.
- **GRPO** — Group Relative Policy Optimization for reasoning and multi-reward
  loops, with a dynamic reward pipeline that mixes custom Python reward
  functions and Hub reward models.

## Design principles

1. **Validate early, fail loudly.** Every hyperparameter lives in a strict
   Pydantic V2 schema (`AlignmentConfig`). A typo'd key or an out-of-range
   `beta` fails at config-load time — before any weights download.
2. **Thin, honest wrappers.** AlignPy trainers translate a validated config
   into the corresponding TRL trainer (targeting TRL >= 1.0 and
   transformers >= 5); they don't reimplement algorithms.
3. **Lazy heavy imports.** `from alignpy import AlignmentConfig` works without
   torch installed; models and datasets only load when training starts.
4. **Rewards are pluggable.** Register any Python callable as a reward
   function and reference it by name from YAML, alongside Hub reward models.

## Project structure

```
align-py/
├── pyproject.toml              # packaging, deps, `alignpy` console script
├── alignpy/
│   ├── __init__.py             # public API (lazy trainer exports)
│   ├── configs.py              # Pydantic V2 schemas: AlignmentConfig + sub-blocks
│   ├── cli.py                  # `alignpy align` / `alignpy eval`
│   ├── rewards.py              # reward-function protocol, registry, built-ins
│   ├── py.typed                # PEP 561 marker — AlignPy ships type hints
│   └── trainers/
│       ├── __init__.py         # build_trainer() dispatch on config.method
│       ├── base.py             # shared lifecycle + RewardLogHook / callback bridge
│       ├── sft.py              # SFTTrainer  (wraps trl.SFTTrainer)
│       ├── dpo.py              # DPOTrainer  (wraps trl.DPOTrainer)
│       └── grpo.py             # GRPOTrainer (wraps trl.GRPOTrainer)
├── examples/
│   ├── dpo_example.py          # full DPO pass in <20 lines
│   └── configs/                # ready-to-run YAML configs (DPO, GRPO)
└── tests/                      # schema/registry/CLI tests (no GPU needed)
```

## Quickstart

```bash
pip install -e .
```

### Python SDK

```python
from alignpy import AlignmentConfig, DPOTrainer

config = AlignmentConfig.from_yaml("examples/configs/dpo_config.yaml")
trainer = DPOTrainer(
    config,
    reward_hooks=[lambda m, step: print(step, m.get("rewards/margins"))],
)
trainer.train()
trainer.save()   # model + tokenizer + alignpy_config.yaml snapshot
```

### CLI

```bash
alignpy align --config examples/configs/dpo_config.yaml
alignpy eval  --model ./qwen-dpo        # reads the saved config snapshot
```

### Custom rewards (GRPO)

```python
from alignpy import register_reward

@register_reward("code_executes")
def code_executes(completions, **kwargs):
    return [1.0 if run_sandboxed(c) else -1.0 for c in completions]
```

```yaml
grpo:
  reward_funcs: [code_executes, length_penalty]
  reward_weights: [1.0, 0.2]
```

## Development

```bash
pip install -e ".[dev]"
ruff check alignpy tests examples   # lint (enforced in CI)
pytest -m "not integration"         # fast lane: schema, reward-registry, CLI, trainer units
pytest tests/                       # + offline end-to-end runs for SFT, DPO, and GRPO,
                                    #   and SmolLM2-135M DPO when the HF Hub is reachable
```

## Releasing

`__version__` in `alignpy/__init__.py` is the single source of truth for the version;
`pyproject.toml` reads it via hatchling. Publishing runs entirely in GitHub Actions
(`.github/workflows/release.yml`) — no credentials are ever needed locally.

```bash
python -m build && twine check --strict dist/*   # optional local pre-flight
```

1. **Dry run**: Actions → Release → Run workflow with *Upload to PyPI* unchecked.
   This builds, runs `twine check --strict`, and publishes the `dist/` artifact for
   inspection — without uploading anything.
2. **Release**: bump `__version__`, then `git tag vX.Y.Z && git push origin vX.Y.Z`.
   The workflow refuses to build if the tag disagrees with `__version__`, and the
   upload waits on approval of the protected `pypi` environment.

A version number is permanent once uploaded — it can never be reused or overwritten,
even after deletion. Run the dry run first when in doubt.


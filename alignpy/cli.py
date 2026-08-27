"""AlignPy command-line interface.

Two entry points, installed as the ``alignpy`` console script::

    alignpy align --config dpo_config.yaml     # run an alignment pass
    alignpy eval  --model ./aligned_model      # evaluate an aligned model

``align`` validates the YAML against :class:`~alignpy.configs.AlignmentConfig`
*before* touching any weights, so schema errors surface in milliseconds.
``eval`` reads the config snapshot AlignPy saves next to every model
(``alignpy_config.yaml``) — or an explicit ``--config`` — rebuilds the matching
trainer with the saved model as the policy, and prints evaluation metrics
(including implicit reward margins for DPO) as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from alignpy.configs import AlignmentConfig

__all__ = ["main"]


def _load_config(path: Path) -> AlignmentConfig:
    """Load + validate a YAML config, exiting with a readable error on failure."""
    try:
        return AlignmentConfig.from_yaml(path)
    except FileNotFoundError:
        sys.exit(f"alignpy: config file not found: {path}")
    except ValidationError as err:
        sys.exit(f"alignpy: invalid config {path}:\n{err}")


def _cmd_align(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if args.validate_only:
        # CI-friendly config linting: schema validation already happened in
        # _load_config, so just echo the fully-resolved config and exit 0.
        print(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
        return
    # Import here so `alignpy align --help` and config validation never need torch.
    from alignpy.trainers import build_trainer

    trainer = build_trainer(config)
    trainer.train()
    output_dir = trainer.save()
    print(f"alignpy: {config.method} run complete; model saved to {output_dir}")


def _cmd_eval(args: argparse.Namespace) -> None:
    model_dir = Path(args.model)
    config_path = args.config or model_dir / "alignpy_config.yaml"
    if not Path(config_path).exists():
        sys.exit(
            f"alignpy: no config snapshot at {config_path}; pass --config explicitly "
            "(models saved via AlignPy include the snapshot automatically)."
        )
    config = _load_config(Path(config_path))
    config.model.model_name_or_path = str(model_dir)  # evaluate the aligned weights
    if config.data.eval_split is None:
        sys.exit("alignpy: eval requires data.eval_split to be set in the config.")
    from alignpy.trainers import build_trainer

    metrics = build_trainer(config).evaluate()
    print(json.dumps(metrics, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """The ``alignpy`` argument parser (exposed separately for testing/docs)."""
    from alignpy import __version__

    parser = argparse.ArgumentParser(
        prog="alignpy",
        description="Lightweight LLM alignment: SFT, DPO, and GRPO on top of TRL.",
    )
    parser.add_argument("--version", action="version", version=f"alignpy {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    align = sub.add_parser("align", help="Run an alignment pass from a YAML config.")
    align.add_argument("--config", type=Path, required=True, help="Path to a YAML config.")
    align.add_argument(
        "--validate-only", action="store_true",
        help="Validate the config and print its resolved form without training.",
    )
    align.set_defaults(func=_cmd_align)

    evaluate = sub.add_parser("eval", help="Evaluate an aligned model.")
    evaluate.add_argument("--model", required=True, help="Path to the aligned model directory.")
    evaluate.add_argument(
        "--config", type=Path, default=None,
        help="Config override; defaults to the snapshot saved inside --model.",
    )
    evaluate.set_defaults(func=_cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

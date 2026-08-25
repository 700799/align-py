"""CLI tests — parsing and config validation paths (no training deps needed)."""

import pytest

from alignpy.cli import build_parser, main


def test_align_requires_config():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["align"])


def test_eval_requires_model():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval"])


def test_align_missing_config_file_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        main(["align", "--config", str(tmp_path / "missing.yaml")])


def test_align_invalid_config_reports_validation_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("method: dpo\nmodel: {model_name_or_path: m}\n"
                   "data: {dataset_name_or_path: d}\ndpo: {beta: -1}\n")
    with pytest.raises(SystemExit, match="invalid config"):
        main(["align", "--config", str(bad)])


def test_eval_without_snapshot_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="no config snapshot"):
        main(["eval", "--model", str(tmp_path)])

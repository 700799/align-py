"""CLI tests — parsing and config validation paths (no training deps needed)."""

import json

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


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "alignpy" in capsys.readouterr().out


def test_align_validate_only_prints_resolved_config(tmp_path, capsys):
    config_file = tmp_path / "ok.yaml"
    config_file.write_text(
        "method: dpo\nmodel: {model_name_or_path: m}\n"
        "data: {dataset_name_or_path: d}\ndpo: {beta: 0.25}\n"
    )
    main(["align", "--config", str(config_file), "--validate-only"])
    out = capsys.readouterr().out
    resolved = json.loads(out)
    assert resolved["method"] == "dpo"
    assert resolved["dpo"]["beta"] == 0.25
    # Defaults were resolved and echoed, e.g. the derived max_length.
    assert resolved["dpo"]["max_length"] == resolved["dpo"]["max_prompt_length"] + \
        resolved["dpo"]["max_completion_length"]


def test_eval_snapshot_found_but_no_eval_split_exits_cleanly(tmp_path):
    (tmp_path / "alignpy_config.yaml").write_text(
        "method: dpo\nmodel: {model_name_or_path: m}\ndata: {dataset_name_or_path: d}\n"
    )
    with pytest.raises(SystemExit, match="eval requires data.eval_split"):
        main(["eval", "--model", str(tmp_path)])

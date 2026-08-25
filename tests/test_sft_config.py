"""``--config`` support on the SFT parser (parity with the RL recipes).

SFT shipped without config-file support while both RL recipes had presets, so
"drive everything from the preset" only worked for two of three recipes. The
loader is shared (``service/config_file.py``) so both paths behave identically.
"""

import json

from training_sessions_cookbook.sft.train import parse_args


def _preset(tmp_path, **overrides):
    cfg = {
        "base_model_arn": "arn:aws:sagemaker:us-west-2:aws:hub-content/X/Model/m/1.0.0",
        "role_arn": "arn:aws:iam::123456789012:role/R",
        "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/g",
        "output_s3_path": "s3://bucket/out/",
        "batch_size": 128,
        "learning_rate": 5e-5,
        "max_steps": 400,
        "git_hash": "abc123",          # meta key: must be ignored, not an error
    }
    cfg.update(overrides)
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def test_config_supplies_values_and_ignores_meta_keys(tmp_path):
    args = parse_args(["--config", _preset(tmp_path)])
    assert args.batch_size == 128
    assert args.learning_rate == 5e-5
    assert args.max_steps == 400
    assert args.role_arn.endswith(":role/R")


def test_explicit_flags_override_the_config(tmp_path):
    args = parse_args(["--config", _preset(tmp_path), "--max_steps", "7",
                       "--batch_size", "16"])
    assert args.max_steps == 7
    assert args.batch_size == 16
    assert args.learning_rate == 5e-5  # untouched keys still come from the file


def test_shipped_presets_load(tmp_path):
    """The recipe's own presets must parse through the real parser."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("sft_tulu3", "sft_ultrachat"):
        path = root / "training_sessions_cookbook/recipes/chat_sft/configs" / f"{name}.json"
        args = parse_args(["--config", str(path)])
        assert args.batch_size > 0 and args.max_steps > 0
        assert args.dataset_path


def test_a_finished_runs_config_is_rerunnable(tmp_path):
    """A run's own config.json must load back through --config.

    That round trip is the point of the feature: reproducing a run means
    pointing at its config, not reconstructing a command line.
    """
    written = {
        "base_model_arn": "arn:aws:sagemaker:us-west-2:aws:hub-content/X/Model/m/1.0.0",
        "role_arn": "arn:aws:iam::123456789012:role/R",
        "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/g",
        "output_s3_path": "s3://bucket/out/",
        "max_steps": 42,
        "git_hash": "deadbeef",
        "unknown_future_key": "tolerated",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(written))
    args = parse_args(["--config", str(path)])
    assert args.max_steps == 42


def test_bad_config_paths_are_clean_cli_errors(tmp_path):
    """A missing or malformed config must exit like an argparse error.

    Previously both surfaced raw FileNotFoundError / JSONDecodeError
    tracebacks - noise for what is simply a mistyped path or a stray comma.
    """
    import pytest

    with pytest.raises(SystemExit):
        parse_args(["--config", str(tmp_path / "nope.json")])

    bad = tmp_path / "bad.json"
    bad.write_text("{ not json ")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(bad)])

    not_object = tmp_path / "list.json"
    not_object.write_text("[1, 2]")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(not_object)])


def test_null_for_a_defaulted_flag_is_rejected(tmp_path):
    """A null where the flag has a real default must fail at parse time.

    Silently accepting it produced ``max_steps=None`` and a confusing
    "NoneType cannot be interpreted as an integer" much later, inside the
    training loop. Nulls for genuinely optional flags (default None) stay
    legal - that is what a run's own config.json contains.
    """
    import pytest

    bad = tmp_path / "null.json"
    bad.write_text(json.dumps({"max_steps": None}))
    with pytest.raises(SystemExit):
        parse_args(["--config", str(bad)])

    # optional flags may be null: this must still parse
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({
        "base_model_arn": "arn:aws:sagemaker:us-west-2:aws:hub-content/X/Model/m/1.0.0",
        "role_arn": "arn:aws:iam::123456789012:role/R",
        "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/g",
        "output_s3_path": "s3://bucket/out/",
        "training_session_name": None,
    }))
    args = parse_args(["--config", str(ok)])
    assert args.training_session_name is None

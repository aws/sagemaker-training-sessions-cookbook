"""Tests for the --config JSON loader (_apply_config_defaults).

The loader lets a run take its hyperparameters from a saved config.json (the
dump_config shape) while explicit CLI flags still override. These tests build a
small argparse parser mirroring the real one's relevant shape and assert:

  * config values become defaults (picked up when the flag is absent);
  * an explicit CLI flag overrides the config value;
  * a required=True arg (like --base_model) is satisfied from the config and its
    `required` is cleared so a config-only invocation parses;
  * unknown keys (e.g. git_hash) are ignored, not errors.

Pure / SDK-free, no external SDK, no backend.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_sessions_cookbook.rl import train as rl  # noqa: E402


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--base_model", required=True)      # required, like the real one
    p.add_argument("--backend", choices=["service"], default="service")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--responses_per_prompt", type=int, default=4)
    p.add_argument("--max_steps_off_policy", type=int, default=0)
    return p


def _write(tmp_path, cfg):
    path = os.path.join(str(tmp_path), "cfg.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def test_config_values_become_defaults(tmp_path):
    p = _parser()
    cfg = _write(tmp_path, {"base_model": "openai/gpt-oss-20b", "batch_size": 128,
                            "responses_per_prompt": 8, "backend": "service"})
    rl._apply_config_defaults(p, cfg)
    args = p.parse_args([])  # nothing on the CLI: all from config
    assert args.base_model == "openai/gpt-oss-20b"
    assert args.batch_size == 128
    assert args.responses_per_prompt == 8
    assert args.backend == "service"


def test_cli_overrides_config(tmp_path):
    p = _parser()
    cfg = _write(tmp_path, {"base_model": "openai/gpt-oss-20b", "batch_size": 128,
                            "backend": "service"})
    rl._apply_config_defaults(p, cfg)
    # Override batch_size + add an off-policy cap not present in the file.
    args = p.parse_args(["--batch_size", "32", "--max_steps_off_policy", "2"])
    assert args.batch_size == 32               # CLI wins
    assert args.backend == "service"           # still from config
    assert args.max_steps_off_policy == 2      # CLI-only


def test_required_arg_satisfied_from_config(tmp_path):
    p = _parser()
    cfg = _write(tmp_path, {"base_model": "openai/gpt-oss-20b"})
    rl._apply_config_defaults(p, cfg)
    # No --base_model on the CLI; must NOT raise (required was cleared).
    args = p.parse_args([])
    assert args.base_model == "openai/gpt-oss-20b"


def test_unknown_keys_ignored(tmp_path):
    p = _parser()
    cfg = _write(tmp_path, {"base_model": "m", "git_hash": "abc123",
                            "some_future_key": 1})
    # Should not raise despite git_hash / some_future_key having no flag.
    rl._apply_config_defaults(p, cfg)
    args = p.parse_args([])
    assert args.base_model == "m"
    assert not hasattr(args, "git_hash")

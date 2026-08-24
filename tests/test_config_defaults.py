"""Parity between the CLI parser and the canonical defaults table.

The runtime reads knobs via ``rl.config.cfg`` whose fallbacks come from
``rl.config.DEFAULTS``. These tests pin the two remaining ways they could
drift apart: an argparse default changing without the table (or vice versa),
and a new hand-typed ``getattr(args, ...)`` fallback sneaking back in.
"""

import pathlib
import re
import subprocess
import sys

from training_sessions_cookbook.rl.config import DEFAULTS, cfg
from training_sessions_cookbook.rl.train import build_parser

import pytest


def test_every_table_knob_matches_its_argparse_default():
    """DEFAULTS and the union parser must agree exactly, knob by knob.

    This is the invariant that was broken before the table existed:
    ``num_rollout_workers`` was 4 on the CLI but 8 at six runtime fallbacks,
    ``code_timeout`` 10 vs 6, ``val_at_start`` True vs False. Which value a
    run used depended on which code path read the knob.
    """
    parser = build_parser()  # all registered envs -> all flags
    argdefs = {a.dest: a.default for a in parser._actions}
    for name, value in DEFAULTS.items():
        assert name in argdefs, f"table knob {name!r} has no CLI flag"
        assert argdefs[name] == value, (
            f"{name!r}: argparse default {argdefs[name]!r} != table {value!r}; "
            "update rl.config.DEFAULTS and the flag together"
        )


def test_cfg_rejects_unknown_knobs():
    class _Args:
        pass

    with pytest.raises(KeyError):
        cfg(_Args(), "definitely_not_a_knob")


def test_cfg_prefers_the_args_value():
    class _Args:
        num_rollout_workers = 99

    assert cfg(_Args(), "num_rollout_workers") == 99
    assert cfg(object(), "num_rollout_workers") == DEFAULTS["num_rollout_workers"]


def test_no_hand_typed_fallbacks_remain():
    """No production code may write ``getattr(args, "knob", <literal>)``.

    ``cfg`` is the only sanctioned read-with-fallback. Allowed exceptions:
    private side-channels (leading underscore, being eliminated separately),
    computed fallbacks that reference another variable, and this config module
    itself.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "training_sessions_cookbook"
    pat = re.compile(r'getattr\(args, "([a-z][a-z_0-9]*)", ')
    offenders = []
    for f in root.rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            m = pat.search(line)
            if not m:
                continue
            name = m.group(1)
            if name in DEFAULTS:
                offenders.append(f"{f.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, (
        "hand-typed fallbacks for table knobs found; use rl.config.cfg:\n"
        + "\n".join(offenders)
    )


def test_no_raw_side_channel_access_outside_runstate():
    """``rl.runstate`` is the only module allowed to touch the private
    run-state attributes on args (``_run_artifacts`` / ``_dataset``).

    Everything else must go through its typed accessors so the contract has
    one implementation and reads cannot silently miss (raw ``getattr`` with a
    default returns the default on a typo; the accessor cannot be typo'd).
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "training_sessions_cookbook"
    pat = re.compile(r'args\._(mlflow|dataset|run_folder|metrics_path|run_artifacts)\b'
                     r'|getattr\(args, "_(mlflow|dataset|run_folder|metrics_path|run_artifacts)"')
    offenders = []
    for f in root.rglob("*.py"):
        if f.name == "runstate.py":
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]  # comments may describe the contract
            if pat.search(code) and '"""' not in line:
                offenders.append(f"{f.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, (
        "raw side-channel access found; use rl.runstate accessors:\n" + "\n".join(offenders)
    )

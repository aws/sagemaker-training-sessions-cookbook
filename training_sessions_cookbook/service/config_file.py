"""``--config`` JSON support, shared by the RL and SFT parsers.

A run's ``config.json`` (written by :class:`service.runlog.RunLog`) is exactly
the shape this loader reads, so any finished run is directly re-runnable::

    python -m training_sessions_cookbook.recipes.chat_sft.train \\
        --config results/<run>/config.json --max_steps 10

Config keys seed argparse *defaults*, so an explicit flag on the command line
always wins over the file. Extracted from the RL parser so both training paths
share one implementation (and one behaviour) instead of drifting.
"""

from __future__ import annotations

import argparse
import json
import logging


def apply_config_defaults(parser: argparse.ArgumentParser, config_path: str) -> None:
    """Seed ``parser`` defaults from a JSON config so CLI flags still override.

    Reads the JSON at ``config_path`` (the shape ``dump_config`` writes) and for
    every key matching a known argparse destination calls
    ``parser.set_defaults(dest=value)``. Because argparse applies an explicitly
    supplied flag *over* its default, any flag on the command line still wins;
    the config only fills in the rest. Two consequences handled here:

    * **Required args become satisfiable from the file.** Once the config
      supplies a ``required=True`` argument, ``required`` is cleared on that
      action so a config-only invocation does not error. If neither the config
      nor the CLI provides it, argparse still errors as before.
    * **Unknown keys are ignored** (with one warning listing them), so a real
      ``config.json`` - which carries ``git_hash`` and any future non-arg keys -
      loads cleanly.

    ``--config`` itself and the meta key ``git_hash`` are never treated as args.
    """
    log = logging.getLogger(__name__)
    # A bad --config path or a typo'd file is a CLI mistake, so report it the
    # way argparse reports one (clean message, exit 2) instead of surfacing a
    # FileNotFoundError / JSONDecodeError traceback at the user.
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"--config {config_path!r}: no such file")
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"--config {config_path!r}: invalid JSON at line {exc.lineno} "
            f"column {exc.colno}: {exc.msg}"
        )
    if not isinstance(cfg, dict):
        raise SystemExit(f"--config {config_path!r} must be a JSON object")

    dest_to_action = {a.dest: a for a in parser._actions}
    ignored = {"config", "git_hash"}

    applied, unknown, nulled = {}, [], []
    for key, value in cfg.items():
        if key in ignored:
            continue
        if key in dest_to_action:
            action = dest_to_action[key]
            # A null where the flag has a real default cannot be expressed on
            # the command line, and silently accepting it defers the failure to
            # a confusing "NoneType cannot be interpreted as an integer" deep in
            # the training loop. Real configs never do this: `dump_config` only
            # writes null where the value WAS None (optional flags), whose
            # defaults are None too - verified across every shipped preset.
            if value is None and action.default is not None:
                nulled.append(key)
                continue
            applied[key] = value
            if getattr(action, "required", False):
                action.required = False
        else:
            unknown.append(key)

    if nulled:
        raise SystemExit(
            f"--config {config_path!r}: null value(s) for flag(s) with a real "
            f"default: {', '.join(sorted(nulled))}. Remove the key to keep the "
            "default, or give it a value."
        )

    if applied:
        parser.set_defaults(**applied)
    if unknown:
        log.warning(
            "--config %s: ignoring %d key(s) with no matching CLI flag: %s",
            config_path, len(unknown), ", ".join(sorted(unknown)),
        )


def parse_known_config_path(argv=None) -> "str | None":
    """Sniff ``--config`` ahead of the real parse.

    A first pass is needed because argparse cannot satisfy ``required=True``
    arguments from a file it has not read yet.
    """
    sniffer = argparse.ArgumentParser(add_help=False)
    sniffer.add_argument("--config", default=None)
    pre_args, _ = sniffer.parse_known_args(argv)
    return pre_args.config

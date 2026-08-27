"""Single source of truth for training-knob defaults.

Every knob that is BOTH registered as a CLI flag and read back at runtime with
a fallback lives here. The parser registers flags with these values, and
:func:`cfg` reads them back, so the CLI default and the runtime fallback can
never diverge.

Why this module exists: the runtime used to read knobs as
``getattr(args, "name", <hand-typed fallback>)`` at ~40 call sites, and several
fallbacks had silently forked from the argparse defaults (``num_rollout_workers``
was 4 on the CLI but 8 at six runtime sites, ``code_timeout`` 10 vs 6,
``val_at_start`` True vs False, ``max_tokens`` 2048 vs None). Which value a run
actually used depended on which code path read the knob first - the worst kind
of configuration bug, invisible until two paths disagree.

The fallbacks are still needed at all because environment flags are
recipe-gated: ``build_parser(envs=("gsm8k",))`` does not register code_rl's
flags, yet shared helpers (validation, dataset loading) probe code_rl knobs on
whatever namespace they were handed. :func:`cfg` keeps that tolerance while
pinning the fallback to the one canonical default.

This module deliberately has no imports from the rest of the package (a leaf),
so both the parser and every runtime consumer can use it without cycles.
"""

from __future__ import annotations

from typing import Any

# Canonical defaults, alphabetical. The values here ARE the argparse defaults;
# ``build_parser`` registers flags from this table.
DEFAULTS: dict[str, Any] = {
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1e-8,
    "advantage_mode": "grouped",
    "answer_format": "hash",
    "clip_high_threshold": None,
    "clip_low_threshold": None,
    "code_timeout": 10,
    "context_overflow_reward": -0.1,
    "dataset_path": None,
    "env_type": "gsm8k",
    "format_coef": 0.1,
    "kl_coef": 0.0,
    "log_every": 10,
    "max_inflight_rollouts": None,
    "max_prompts": None,
    "max_steps_off_policy": 0,
    "max_tokens": 2048,
    "max_trajectory_tokens": 32768,
    "mlflow_traces_per_step": 0,
    "num_rollout_workers": 4,
    "record_trajectories": False,
    "region": "us-west-2",
    "rolling_save_every": 10,
    "sampling_seed": None,
    "save_weights_at_end": False,
    "seed": 0,
    "stale_policy": "drop",
    "stop": None,
    "task_offset": 0,
    "top_k": None,
    "top_p": None,
    "training_session_name": None,
    "val_at_start": True,
    "val_code_split": "none",
    "val_dataset_path": None,
    "val_max_prompts": None,
    "warmup_steps": 0,
}


def cfg(args: Any, name: str) -> Any:
    """Read knob ``name`` from ``args``, falling back to its canonical default.

    The only sanctioned replacement for ``getattr(args, name, <literal>)``:
    the fallback comes from :data:`DEFAULTS`, so it is always the same value
    the CLI would have supplied. Unknown names raise ``KeyError`` immediately -
    a typo in a knob name is a bug, not a knob that happens to be unset.
    """
    if name not in DEFAULTS:
        raise KeyError(
            f"unknown config knob {name!r}; add it to rl.config.DEFAULTS "
            "if it is a real knob"
        )
    return getattr(args, name, DEFAULTS[name])


def default(name: str) -> Any:
    """The canonical default for ``name``, for ``add_argument(default=...)``."""
    if name not in DEFAULTS:
        raise KeyError(f"unknown config knob {name!r}")
    return DEFAULTS[name]

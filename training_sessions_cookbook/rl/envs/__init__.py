"""Pluggable RL environment package.

Environments are looked up by name in a registry. The two built-ins (GSM8K,
code RL) are registered below; add your own with :func:`register_env` and the
whole training loop, rollouts, advantages, off-by-k, metrics, checkpointing
-- works unchanged::

    from training_sessions_cookbook.rl import envs

    class MyEnv:                      # implements the Env protocol (base.py)
        ...

    envs.register_env("my_task", lambda args: (lambda: MyEnv(args.dataset_path)))
    # now: build_parser(envs=("my_task",)) / --env_type my_task

Concrete env modules and their heavy dependencies (``datasets``, ``aiohttp``,
...) are imported lazily inside the built-in factory builders, so importing
this package, or building a parser, never pulls in the optional
``code-rl`` extras.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .base import Env, Message, ToolCall, Transition
from ..config import cfg
from ..runstate import shared_dataset

__all__ = [
    "Env",
    "Message",
    "ToolCall",
    "Transition",
    "get_env_factory",
    "register_env",
    "registered_envs",
    "env_cli_args",
]


@dataclass(frozen=True)
class _EnvSpec:
    #: (args) -> zero-arg factory producing fresh Env instances.
    factory_builder: Callable[[object], Callable[[], "Env"]]
    #: Optional (parser) -> None hook adding this env's CLI flags.
    add_cli_args: Optional[Callable] = None


_REGISTRY: dict[str, _EnvSpec] = {}


def register_env(name: str, factory_builder, add_cli_args=None) -> None:
    """Register an environment under ``name`` (becomes a valid ``--env_type``).

    Args:
        name: The ``--env_type`` value selecting this environment.
        factory_builder: ``(args) -> factory`` where ``factory()`` returns a
            fresh object implementing the :class:`~.base.Env` protocol. Read
            any flags you need off ``args``.
        add_cli_args: Optional ``(parser) -> None`` adding the environment's
            own CLI flags; included when a parser is built with this env.
    """
    _REGISTRY[name] = _EnvSpec(factory_builder, add_cli_args)


def registered_envs() -> tuple[str, ...]:
    """Names of all registered environments (built-ins + custom)."""
    return tuple(_REGISTRY)


def env_cli_args(name: str):
    """The env's CLI hook, or ``None``. Raises ``ValueError`` when unknown."""
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown env_type: {name!r} (registered: {sorted(_REGISTRY)})"
        )
    return _REGISTRY[name].add_cli_args


def get_env_factory(env_type: str, args) -> Callable[[], "Env"]:
    """Resolve an ``env_type`` string to a zero-arg env factory callable.

    Args:
        env_type: The ``--env_type`` value selecting the environment.
        args: The parsed CLI args namespace; env factories read the flags
            they need off of it (e.g. ``args.max_turns``).

    Returns:
        A zero-argument callable that constructs a fresh ``Env`` instance.

    Raises:
        ValueError: If ``env_type`` is not registered.
    """
    if env_type not in _REGISTRY:
        raise ValueError(
            f"unknown env_type: {env_type!r} (registered: {sorted(_REGISTRY)})"
        )
    return _REGISTRY[env_type].factory_builder(args)


# ---------------------------------------------------------------------------
# Built-ins. Factory builders import their env modules lazily; the CLI hooks
# are defined here (stdlib only) so parser construction stays dependency-light.
# ---------------------------------------------------------------------------


def _gsm8k_factory_builder(args):
    from .gsm8k import GSM8KMultiTurnEnv

    return lambda: GSM8KMultiTurnEnv(
        max_turns=args.max_turns,
        dataset=shared_dataset(args),
        answer_format=cfg(args, "answer_format"),
    )


def _gsm8k_cli_args(parser) -> None:
    g = parser.add_argument_group("gsm8k")
    g.add_argument(
        "--answer_format",
        choices=["hash", "boxed", "tagged", "the_answer_is", "auto"],
        default="hash",
        help="GSM8K answer-format knob (controls both the instruction text "
             "appended to the user prompt and the reward extractor's regex "
             "set). `hash` is the default: "
             '"output the final answer after \\"####\\"" + regex `#### N`. '
             "`boxed`/`tagged`/`the_answer_is` instruct the model to use a "
             "different marker. `auto` drops the format hint from the prompt "
             "and lets the extractor try every known marker. Two runs that "
             "differ only in --answer_format are NOT directly comparable on "
             "reward (different task contract), but accuracy is.",
    )


def _code_rl_factory_builder(args):
    from .code_rl import build_code_rl_env_factory

    return build_code_rl_env_factory(args)


def _code_rl_cli_args(parser) -> None:
    # Consumed by build_code_rl_env_factory via getattr.
    g = parser.add_argument_group("code_rl")
    g.add_argument("--code_timeout", type=int, default=10)
    g.add_argument("--format_coef", type=float, default=0.1)
    g.add_argument("--context_overflow_reward", type=float, default=-0.1)
    g.add_argument("--max_trajectory_tokens", type=int, default=32768)
    g.add_argument("--seed", type=int, default=0)


register_env("gsm8k", _gsm8k_factory_builder, _gsm8k_cli_args)
register_env("code_rl", _code_rl_factory_builder, _code_rl_cli_args)

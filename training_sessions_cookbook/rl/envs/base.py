"""Shared environment types and the Env protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass
class Transition:
    """One turn of one rollout episode.

    `obs_tokens` captures the running prompt + history tokens at this turn; `act_tokens` / `act_logprobs` are the sampled assistant tokens and the per-token logprobs returned by the sampler. `advantage` is populated later by the advantage helpers; it defaults to `0.0` so freshly-built transitions are immediately consumable by `build_training_datum` even before advantages are filled in.
    """

    obs_tokens: list[int]
    act_tokens: list[int]
    act_logprobs: list[float]
    reward: float
    done: bool
    env_type: str
    advantage: float = 0.0


@dataclass
class ToolCall:
    """A structured tool/function call parsed from model output."""

    name: str
    arguments: dict


@dataclass
class Message:
    """A single chat message in a (possibly multi-turn) trajectory.

    `role` is one of "system" | "user" | "assistant" | "tool". `tool_calls` is populated for assistant messages that invoke tools.
    """

    role: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@runtime_checkable
class Env(Protocol):
    """Duck-typed environment interface consumed by the training loop.

    Concrete envs (e.g. ``GSM8KMultiTurnEnv``, ``CodeRLEnv``) implement these methods. The protocol is ``runtime_checkable`` so tests can assert that a factory product conforms via ``isinstance(env, Env)``.
    """

    def assign_problem(self, *args, **kwargs) -> None:
        """Assign the next problem/sample to this env instance."""
        ...

    def reset(self, seed: Optional[int] = None) -> tuple[str, dict]:
        """Reset the episode and return ``(observation, info)``."""
        ...

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """Advance one turn, returning ``(obs, reward, terminated, truncated, info)``.
        """
        ...

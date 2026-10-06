"""Advantage computation: Monte-Carlo (global baseline) and grouped (per-prompt)."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rollouts import Transition


# === Advantage helpers ===
#
# Implements the two advantage-computation helpers used by the training loop:
#
#   - compute_monte_carlo_advantages(episodes, gamma, use_rebn) applies the discounted-return recurrence G_T = r_T, G_t = r_t + gamma * G_{t+1} to each episode independently, then optionally ReB-normalizes the resulting advantage vector (mean-zero, unit-variance) across all transitions.
#
#   - compute_grouped_advantages(groups) applies the GRPO-style per-group centering (return - mean) / (std + eps) across the K episodes that share a prompt, broadcasting the per-episode scalar advantage onto every transition in that episode.
#
# Both helpers return a flat list of Transition (with `transition.advantage` populated) so the training loop can build datums in one pass.
#
# The `Transition` dataclass is defined in `rollouts.py`; here we only rely on its `reward` and `advantage` attributes via duck typing, and use string annotations to forward-reference the type.

_ADV_EPS = 1e-8


def advantage_values(transition: "Transition") -> list[float]:
    """A transition's advantage as a flat list: ``[scalar]`` or the per-token list.

    Advantages are a scalar per transition for RL (broadcast over every action
    token at datum-build time) and a per-token list for on-policy distillation.
    Metrics use this to treat both shapes uniformly.
    """
    adv = transition.advantage
    if isinstance(adv, (list, tuple)):
        return [float(a) for a in adv]
    return [float(adv)]


def mean_advantage(transition: "Transition") -> float:
    """Scalar summary of a transition's advantage (the per-token mean when a list)."""
    values = advantage_values(transition)
    return sum(values) / len(values) if values else 0.0


def compute_monte_carlo_advantages(
    episodes: "list[list[Transition]]",
    gamma: float,
    use_rebn: bool,
) -> "list[Transition]":
    """Compute Monte-Carlo discounted returns and write them as advantages.

    For each episode of length T, the discounted return is computed by reverse iteration:

        G[T-1] = r[T-1]
        G[t]   = r[t] + gamma * G[t+1]   for t < T-1

    Each transition's `advantage` is set to its own G[t]. When `use_rebn` is True the advantages are then renormalized across *all* transitions in the batch to mean 0 and standard deviation 1 (with a small epsilon guarding against zero variance).

    Args:
        episodes: A list of episodes, where each episode is a list of Transition objects in time order.
        gamma: Discount factor in [0, 1].
        use_rebn: If True, apply ReB normalization across the flat batch.

    Returns:
        A flat list of Transition with `transition.advantage` populated.
    """
    flat: list = []
    for episode in episodes:
        if not episode:
            continue
        # Reverse-iteration recurrence: G[T-1] = r[T-1]; G[t] = r[t] + gamma * G[t+1].
        running = 0.0
        for transition in reversed(episode):
            running = float(transition.reward) + gamma * running
            transition.advantage = running
            flat.append(transition)

    if use_rebn and len(flat) > 0:
        adv = np.asarray([t.advantage for t in flat], dtype=np.float64)
        mean = float(adv.mean())
        std = float(adv.std())
        normalized = (adv - mean) / (std + _ADV_EPS)
        for transition, value in zip(flat, normalized):
            transition.advantage = float(value)

    return flat


def compute_grouped_advantages(
    groups: "list[list[list[Transition]]]",
) -> "list[Transition]":
    """Compute GRPO-style per-group advantages and broadcast onto transitions.

    Each group is the K episodes generated for a single prompt. For each group we:

      1. Compute per-episode returns as the sum of per-transition rewards.
      2. Center and scale across the K returns:
             advantage = (return - group_mean) / (group_std + eps)
      3. Broadcast that scalar advantage onto every transition in the episode.

    Args:
        groups: A list of groups, where each group is a list of K episodes and each episode is a list of Transition objects.

    Returns:
        A flat list of Transition (across all groups and episodes) with `transition.advantage` populated.
    """
    flat: list = []
    for group in groups:
        if not group:
            continue
        returns = np.asarray(
            [sum(float(t.reward) for t in episode) for episode in group],
            dtype=np.float64,
        )
        group_mean = float(returns.mean())
        group_std = float(returns.std())
        centered = (returns - group_mean) / (group_std + _ADV_EPS)

        for episode, episode_advantage in zip(group, centered):
            scalar = float(episode_advantage)
            for transition in episode:
                transition.advantage = scalar
                flat.append(transition)

    return flat


"""Learning-rate scheduling shared by the RL and SFT training loops.

A step's effective LR is ``base_lr * warmup_factor * decay_multiplier``:

* **warmup** ramps linearly from 0 to 1 over the first ``warmup_steps`` steps, then stays at 1. RL uses this; SFT leaves ``warmup_steps=0`` (factor 1).
* **decay** shapes the LR across the whole run: ``linear`` and ``cosine`` anneal toward 0, ``constant`` holds. SFT uses this; RL leaves it ``constant`` (multiplier 1).
"""

from __future__ import annotations

import math

_MIN_MULTIPLIER = 1e-8


def warmup_factor(step: int, warmup_steps: int) -> float:
    """Linear warmup ramp in [0, 1]; 1.0 when ``warmup_steps <= 0``."""
    if warmup_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0


def decay_multiplier(schedule: str, step: int, total_steps: int) -> float:
    """Decay multiplier in (0, 1] for the given schedule.

    ``total_steps <= 1`` (or ``constant``) yields 1.0. ``linear`` and ``cosine`` anneal over ``progress = step / (total_steps - 1)``, floored at a small positive value so the LR never reaches exactly zero.
    """
    if schedule == "constant" or total_steps <= 1:
        return 1.0
    progress = step / (total_steps - 1)
    if schedule == "linear":
        return max(_MIN_MULTIPLIER, 1.0 - progress)
    if schedule == "cosine":
        return max(_MIN_MULTIPLIER, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return 1.0


def learning_rate(
    base_lr: float,
    step: int,
    *,
    total_steps: int = 0,
    schedule: str = "constant",
    warmup_steps: int = 0,
) -> float:
    """Effective LR for ``step``: base scaled by warmup then decay."""
    return (
        base_lr
        * warmup_factor(step, warmup_steps)
        * decay_multiplier(schedule, step, total_steps)
    )

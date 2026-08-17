"""Bounded-staleness plumbing for the overlapped (off-by-k) training loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rollouts import WrappedGroup



class _AsyncLoopState:
    """Shared mutable state between the consumer and the rollout workers.

    Single-threaded asyncio means attribute reads/writes are atomic *between*
    awaits, so a worker that reads ``step`` and ``sampling_client`` back-to-back
    (no await between) always sees a consistent (step, client) pair, and the
    consumer flips both without an await between them on refresh. That is what
    makes the off-by-k stamping race-free without a lock.
    """

    __slots__ = ("step", "sampling_client")

    def __init__(self) -> None:
        self.step = 0
        self.sampling_client = None


# Sentinels for the shutdown cascade. Identity-compared, never trained on.
_PROMPT_SENTINEL = object()  # dataloader → worker: "no more prompts"
_WORKER_DONE = object()      # worker → consumer: "this producer has stopped"


class _StalenessBatcher:
    """Bounded-staleness filter + accumulate-to-``batch_size`` for off-by-k.

    Encapsulates the three decisions the training consumer makes per group so
    they are unit-testable independent of the async plumbing:

    * **staleness filter**: a group whose ``current_step - birth_step`` exceeds
      ``max_steps_off_policy`` is too off-policy for the IS ratio to correct.
      Under ``stale_policy='drop'`` (default), the group is discarded and its
      prompt is lost from the epoch cursor (counted in ``num_dropped``). Under
      ``stale_policy='requeue'``, the caller is handed back the prompt via the
      ``stale_prompt_to_requeue`` return field so it can be pushed back onto
      ``prompt_queue`` for a fresh on-policy resample (counted in ``num_requeued``).
    * **accumulate to batch_size**: fresh-enough groups are collected until
      ``batch_size`` of them are held, at which point they are released as one
      batch and the accumulator resets. This keeps the trained batch a constant
      ``batch_size`` groups even as stale groups are dropped (the
      groups-per-batch role, here reusing ``--batch_size``).

    ``offer(group, current_step)`` returns a 2-tuple ``(batch, stale_prompt)``:
      * ``batch`` is the full ``list[WrappedGroup]`` when ``batch_size`` fresh
        groups have accumulated, else ``None``.
      * ``stale_prompt`` is the prompt to requeue when ``stale_policy='requeue'``
        and the just-offered group was over-stale, else ``None``.
    """

    __slots__ = (
        "batch_size", "max_steps_off_policy", "stale_policy",
        "_accumulated", "num_dropped", "num_requeued",
    )

    def __init__(
        self,
        batch_size: int,
        max_steps_off_policy: int,
        stale_policy: str = "drop",
    ) -> None:
        if stale_policy not in ("drop", "requeue"):
            raise ValueError(
                f"unknown stale_policy {stale_policy!r}; expected 'drop' or 'requeue'"
            )
        self.batch_size = batch_size
        self.max_steps_off_policy = max_steps_off_policy
        self.stale_policy = stale_policy
        self._accumulated: "list[WrappedGroup]" = []
        self.num_dropped = 0
        self.num_requeued = 0

    def offer(self, group: "WrappedGroup", current_step: int):
        """Classify one group; return ``(batch_or_None, stale_prompt_or_None)``."""
        staleness = current_step - group.birth_step
        if staleness > self.max_steps_off_policy:
            if self.stale_policy == "requeue":
                # Return the stale prompt so the caller can put it back onto
                # prompt_queue. The rollout compute is still wasted (already
                # spent), but the prompt is not lost from the epoch cursor.
                self.num_requeued += 1
                return None, group.prompt
            self.num_dropped += 1
            return None, None
        self._accumulated.append(group)
        if len(self._accumulated) < self.batch_size:
            return None, None
        batch = self._accumulated
        self._accumulated = []
        return batch, None

    @property
    def pending(self) -> "list[WrappedGroup]":
        """Groups accumulated so far but not yet released as a full batch."""
        return list(self._accumulated)



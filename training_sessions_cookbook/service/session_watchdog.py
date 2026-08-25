"""Session-liveness watchdog, shared by the RL and SFT loops.

The service can mark a session Failed server-side (engine fault,
checkpoint-load failure) while the client is awaiting operation results that
will now never resolve. Nothing on the operation path surfaces the session's
death, so without a watchdog a training loop hangs silently forever (observed
on prod, 2026-08-24: an inference engine failed to load a sampler checkpoint;
the run went quiet for 80 minutes until killed by hand).

The watchdog polls ``DescribeTrainingSession`` on a slow cadence and, once the
session leaves the live states, fails every guarded wait loudly with the
service's own ``FailureReason``.

Usage::

    watchdog = SessionWatchdog(svc, session_name)
    result = await watchdog.guard(op.result_async(), what="forward_backward")
    ...
    watchdog.stop()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Session statuses under which operations can still complete.
LIVE_STATUSES = ("Pending", "Created", "Running", "Blocked")


class SessionDeadError(RuntimeError):
    """The training session is terminal server-side; no operation can succeed.

    A DISTINCT type on purpose: the rollout layer treats generic exceptions as
    droppable per-sample failures (warn, requeue, carry on), which is right
    for a flaky sample but fatal to miss for a dead session - the loop would
    churn forever resubmitting work that can never complete (observed
    2026-08-25: a failed session became a two-hour retry storm that ended in
    the client being OOM-killed). Best-effort handlers must let this type
    propagate.
    """


class SessionWatchdog:
    """Fails guarded waits fast when the session dies server-side."""

    def __init__(self, svc: Any, session_name: str, *, interval: float = 60.0) -> None:
        self._svc = svc
        self._session_name = session_name
        self._interval = interval
        self._task: "Optional[asyncio.Task]" = None
        self._stopped = False
        self._loop_ref = None
        self._dead: "Optional[asyncio.Event]" = None
        self.failure: "Optional[str]" = None

    # -- polling ------------------------------------------------------------

    def _read_status(self) -> "tuple[str, str]":
        """Return ``(status, failure_reason)`` from DescribeTrainingSession."""
        resp = self._svc._sagemaker_client.describe_training_session(
            TrainingSessionName=self._session_name
        )
        summary = resp.get("TrainingSessionDetail", resp)
        status = str(summary.get("TrainingSessionStatus", summary.get("Status", "")))
        reason = str(
            summary.get("FailureReason")
            or summary.get("ErrorMessage")
            or summary.get("ErrorDetails")
            or ""
        )
        return status, reason

    async def _run(self) -> None:
        consecutive_failures = 0
        polls = 0
        while True:
            await asyncio.sleep(self._interval)
            polls += 1
            try:
                status, reason = await asyncio.to_thread(self._read_status)
            except Exception as exc:
                # A failed poll is not a dead session - but a BLIND watchdog
                # must be visible, not silent: under an account-wide throttle
                # the describes can fail exactly when the session is most
                # likely dying (observed 2026-08-25, a 429 storm).
                consecutive_failures += 1
                if consecutive_failures % 5 == 0:
                    logger.warning(
                        "session watchdog: %d consecutive describe failures "
                        "(last: %r) - session liveness is UNKNOWN",
                        consecutive_failures, exc,
                    )
                continue
            consecutive_failures = 0
            if polls % 10 == 0:
                logger.info("session watchdog: session %r is %s (poll %d)",
                            self._session_name, status or "?", polls)
            if status and status not in LIVE_STATUSES:
                self.failure = (
                    f"training session {self._session_name!r} is {status} "
                    f"server-side: {reason or 'no reason given'}"
                )
                logger.error("session watchdog: %s", self.failure)
                self._dead.set()
                return

    def _ensure_started(self) -> None:
        """Start (or restart) the watchdog on the current loop; idempotent."""
        if self._stopped:
            raise RuntimeError(
                "SessionWatchdog.guard() called after stop(); the watchdog is "
                "terminal - construct a new one for a new run"
            )
        loop = asyncio.get_running_loop()
        if self._loop_ref is not loop:
            self._loop_ref = loop
            self._dead = asyncio.Event()
            self._task = loop.create_task(self._run(), name="session-watchdog")

    # -- public surface -------------------------------------------------------

    async def guard(self, awaitable: Any, *, what: str) -> Any:
        """Await an operation, failing fast if the session dies server-side.

        Races the operation against the death event; on death the pending
        operation is cancelled and a ``RuntimeError`` carrying the service's
        FailureReason is raised - the alternative is an unbounded wait on a
        result that can never arrive.
        """
        self._ensure_started()
        op_task = asyncio.ensure_future(awaitable)
        death = asyncio.ensure_future(self._dead.wait())
        done, _ = await asyncio.wait({op_task, death}, return_when=asyncio.FIRST_COMPLETED)
        # If both complete in the same tick, prefer the operation's real result
        # (or its real exception) over the death signal - branch order matters.
        if op_task in done:
            death.cancel()
            return op_task.result()
        op_task.cancel()
        raise SessionDeadError(f"{what} abandoned: {self.failure}")

    def stop(self) -> None:
        """Cancel the poller (teardown); terminal, safe to call repeatedly."""
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            self._task = None


class NullWatchdog:
    """No-op stand-in: guards degrade to plain awaits, stop() does nothing.

    For callers that cannot (or choose not to) poll session status - e.g.
    programmatic embeddings of the SFT loop that pass ``watchdog=None``.
    """

    failure: "Optional[str]" = None

    async def guard(self, awaitable: Any, *, what: str) -> Any:
        return await awaitable

    def stop(self) -> None:
        pass


def result_with_liveness(
    future: Any,
    svc: Any,
    session_name: str,
    *,
    timeout: "Optional[float]" = None,
    poll_interval: float = 30.0,
) -> Any:
    """Synchronous ``future.result()`` that fails fast on session death.

    For code that runs before an event loop exists (session setup): waits in
    ``poll_interval`` slices, checking DescribeTrainingSession between slices.
    A dead session raises ``RuntimeError`` with the service's FailureReason
    instead of blocking forever while the session bills. ``timeout`` bounds the
    TOTAL wait (``None`` = unbounded, matching the op-timeout defaults).
    """
    import time as _time

    watch = SessionWatchdog(svc, session_name)
    deadline = None if timeout is None else _time.monotonic() + timeout
    while True:
        slice_ = poll_interval
        if deadline is not None:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for result on {session_name!r}")
            slice_ = min(slice_, remaining)
        try:
            return future.result(timeout=slice_)
        except TimeoutError:
            pass
        except Exception as exc:
            # SDK futures may raise their own timeout type; only keep waiting
            # for timeout-shaped errors, propagate everything else.
            if "imeout" not in type(exc).__name__ and "imeout" not in str(exc):
                raise
        try:
            status, reason = watch._read_status()
        except Exception:
            continue  # a failed poll is not a dead session
        if status and status not in LIVE_STATUSES:
            raise RuntimeError(
                f"session {session_name!r} is {status} server-side while waiting "
                f"for initial save: {reason or 'no reason given'}"
            )

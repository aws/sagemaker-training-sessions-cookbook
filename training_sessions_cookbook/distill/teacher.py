"""Teacher clients: where the teacher's per-token log-probs come from.

Every teacher is a sampling client scored with ``compute_logprobs_async``,
which runs one prefill over a sequence and returns one log-prob per token
(``None`` for the first). They differ only in which weights the client serves:

=============  ====================================================  ============
``--teacher``  Weights                                               Sessions
=============  ====================================================  ============
``base``       the student's frozen base model (ARN-less client)     student's
``checkpoint`` a saved adapter of the student's base model           + 1 teacher
``lagged``     a snapshot of the student, re-bound every N steps     + 1 teacher
``session``    a different base model                                + 1 teacher
=============  ====================================================  ============

Why the non-base teachers get their own session: a session has a single
bound sampling path. A non-base checkpoint must be bound with
``load_weights_for_sampling`` before it can be sampled (the pattern
``recipes/chat_sft/analyze.py`` uses), and the student's per-step sampler
refresh re-binds that path to its newest checkpoint. A teacher checkpoint in
the student's session would be unbound on the next refresh. ARN-less requests
always take the base-model branch, which is why ``base`` can share the
student's session (as KTO's reference client does).

A teacher session only ever samples (no ``forward_backward`` /
``optim_step``). It is billed like any session, so it is stopped and deleted in
:meth:`SessionTeacher.close`. The teacher must use the student's tokenizer: the
student's token IDs are scored as-is (see :mod:`.context`), so this is checked
when the session opens.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Optional, Sequence

from training_sessions_cookbook.service import open_session
from training_sessions_cookbook.service.session_watchdog import NullWatchdog, SessionWatchdog

logger = logging.getLogger(__name__)

TEACHER_SOURCES = ("base", "checkpoint", "lagged", "session")


async def compute_logprobs(client: Any, tokens: list[int]) -> list[Optional[float]]:
    """``client.compute_logprobs_async(prompt=tokens)`` as a plain list.

    Tolerates the SDK shapes in use: the call may return the result directly
    or an awaitable/future (``result_async``), and the result may be the list
    itself or a ``ComputeLogprobsResult`` carrying it in ``.logprobs``.
    """
    out = client.compute_logprobs_async(prompt=tokens)
    if inspect.isawaitable(out):
        out = await out
    if hasattr(out, "result_async"):
        out = await out.result_async()
    logprobs = getattr(out, "logprobs", out)
    return list(logprobs)


class Teacher:
    """Scores token sequences under one sampling client.

    ``score`` returns the teacher's log-probs for the last ``n_tail`` tokens of
    ``tokens``. Requests are fired concurrently (bounded by
    ``max_concurrency``) so the inference server can batch them.
    """

    def __init__(self, client: Any, *, name: str, max_concurrency: int = 32,
                 watchdog: Any = None) -> None:
        self.client = client
        self.name = name
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._watchdog = watchdog or NullWatchdog()

    async def score(self, tokens: Sequence[int], n_tail: int) -> list[Optional[float]]:
        if n_tail <= 0:
            return []
        if n_tail >= len(tokens):
            raise ValueError(
                f"teacher sequence of {len(tokens)} tokens cannot score a tail of "
                f"{n_tail}: the first token has no log-prob"
            )
        async with self._sem:
            logprobs = await self._watchdog.guard(
                compute_logprobs(self.client, list(tokens)),
                what=f"teacher scoring ({self.name})",
            )
        if len(logprobs) != len(tokens):
            raise RuntimeError(
                f"compute_logprobs returned {len(logprobs)} values for {len(tokens)} tokens"
            )
        return logprobs[len(tokens) - n_tail:]

    async def update(self, step: int, sampling_client: Any) -> None:
        """Hook awaited before each scoring call with the student's current sampler."""

    def close(self) -> None:
        """Release resources the teacher owns (nothing, for a borrowed client)."""


class SessionTeacher(Teacher):
    """A teacher served by its own Training Session (stopped and deleted on close)."""

    def __init__(self, client: Any, *, svc: Any, session_name: str,
                 training_client: Any = None, **kwargs) -> None:
        super().__init__(client, **kwargs)
        self._svc = svc
        self._session_name = session_name
        self._tc = training_client
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        watchdog = self._watchdog
        if hasattr(watchdog, "stop"):
            watchdog.stop()
        try:
            self._svc.delete_session(self._session_name, stop_if_active=True)
            logger.info("Teacher session deleted: %s", self._session_name)
        except Exception as exc:  # best-effort teardown
            logger.warning("Could not delete teacher session %s: %r", self._session_name, exc)


class LaggedSelfTeacher(SessionTeacher):
    """The student itself, re-snapshotted every ``refresh_every`` steps.

    SDPO needs a slowly moving self-teacher: grading with the current weights
    diverges (Hubotter et al., 2026). The service has no EMA adapter, so a
    periodic snapshot approximates one. Runs in its own session on the
    student's base model: it starts at the base (step 0), and every
    ``refresh_every`` steps binds that session's sampling path to the
    student's latest sampler checkpoint (``sampling_client.model_package_arn``).
    """

    def __init__(self, client: Any, *, refresh_every: int, **kwargs) -> None:
        super().__init__(client, **kwargs)
        self.refresh_every = max(1, int(refresh_every))
        self.snapshot_step = 0

    async def update(self, step: int, sampling_client: Any) -> None:
        if step <= 0 or step % self.refresh_every or step == self.snapshot_step:
            return
        arn = getattr(sampling_client, "model_package_arn", "") or ""
        if not arn:
            logger.warning("lagged teacher: step-%d sampler has no checkpoint ARN; "
                           "keeping the step-%d snapshot", step, self.snapshot_step)
            return
        self.snapshot_step = step  # set first: concurrent callers skip the reload
        await self._watchdog.guard(
            asyncio.to_thread(self._tc.load_weights_for_sampling, arn),
            what=f"lagged teacher load (step {step})",
        )
        self.client = self._tc.create_sampling_client(model_package_arn=arn)
        logger.info("SDPO teacher re-bound to the step-%d policy: %s", step, arn)


def check_tokenizers_match(student_tokenizer, teacher_tokenizer) -> Optional[str]:
    """``None`` when the two tokenizers agree, else a description of the mismatch.

    Compares vocabulary size and the encoding of a probe string covering
    words, digits, code and whitespace. Not a proof of identity, but it catches
    cross-family pairs, whose token IDs would be scored as unrelated text.
    """
    probe = "Natalia sold 48 clips; def f(x):\n    return x ** 2  # 3.14 <think>"
    sizes = [len(t) if hasattr(t, "__len__") else getattr(t, "vocab_size", None)
             for t in (student_tokenizer, teacher_tokenizer)]
    if None not in sizes and sizes[0] != sizes[1]:
        return f"vocabulary sizes differ (student {sizes[0]}, teacher {sizes[1]})"
    s_ids = list(student_tokenizer.encode(probe))
    t_ids = list(teacher_tokenizer.encode(probe))
    if s_ids != t_ids:
        return f"probe string encodes differently ({len(s_ids)} vs {len(t_ids)} tokens)"
    return None


def _open_teacher_session(args, student_tokenizer, *, base_model_arn,
                          model_package_arn=None):
    """Open a sampling-only teacher session; return ``(svc, name, tc, client)``.

    With ``model_package_arn`` the session's sampling path is bound to that
    adapter and the client samples it; otherwise the client is ARN-less (the
    session's base weights).
    """
    session_name = f"{args.run_name}-teacher-{int(time.time())}"[:63]
    svc, tc, tokenizer = open_session(
        session_name=session_name,
        base_model_arn=base_model_arn,
        model_package_group_arn=args.model_package_group_arn,
        role_arn=args.role_arn,
        output_s3_path=args.output_s3_path,
        # The teacher is never trained; the adapter just has to exist.
        lora_rank=args.teacher_lora_rank,
        lora_alpha=args.teacher_lora_rank,
        region=args.region,
        user_metadata={"run_name": str(args.run_name), "role": "distillation-teacher"},
    )
    try:
        mismatch = check_tokenizers_match(student_tokenizer, tokenizer)
        if mismatch and not args.allow_tokenizer_mismatch:
            raise SystemExit(
                f"teacher tokenizer does not match the student's: {mismatch}. "
                "On-policy distillation scores the student's token IDs under the "
                "teacher, so both must share a tokenizer (use a same-family "
                "teacher, or pass --allow_tokenizer_mismatch to override)."
            )
        if mismatch:
            logger.warning("Teacher tokenizer mismatch ignored (--allow_tokenizer_mismatch): %s", mismatch)
        if model_package_arn:
            # Bind the adapter, then sample it (recipes/chat_sft/analyze.py pattern).
            tc.load_weights_for_sampling(model_package_arn)
            client = tc.create_sampling_client(model_package_arn=model_package_arn)
        else:
            # ARN-less client = the teacher's base weights (its adapter is zero).
            client = tc.create_sampling_client()
    except BaseException:
        try:
            svc.delete_session(session_name, stop_if_active=True)
        except Exception as exc:
            logger.warning("Could not delete teacher session %s after setup failure: %r",
                           session_name, exc)
        raise
    return svc, session_name, tc, client


async def make_teacher(args, backend, tokenizer) -> Teacher:
    """Build the teacher selected by ``--teacher`` (see the module table)."""
    source = args.teacher
    concurrency = args.teacher_max_concurrency
    if source == "base":
        # Shares the student's session: ARN-less requests take the base branch.
        client = backend.training_client.create_sampling_client()
        return Teacher(client, name="base", max_concurrency=concurrency,
                       watchdog=getattr(backend, "watchdog", None))
    if source not in ("checkpoint", "lagged", "session"):
        raise ValueError(f"unknown --teacher {source!r}; expected one of {TEACHER_SOURCES}")

    base_model_arn = (args.teacher_base_model_arn if source == "session"
                      else args.base_model_arn)
    model_package_arn = args.teacher_model_package_arn if source == "checkpoint" else None
    logger.info("Opening %s teacher session on %s%s", source, base_model_arn,
                f" (adapter {model_package_arn})" if model_package_arn else "")
    svc, session_name, tc, client = await asyncio.to_thread(
        _open_teacher_session, args, tokenizer,
        base_model_arn=base_model_arn, model_package_arn=model_package_arn)
    common = dict(svc=svc, session_name=session_name, training_client=tc,
                  max_concurrency=concurrency, watchdog=SessionWatchdog(svc, session_name))
    if source == "lagged":
        return LaggedSelfTeacher(client, refresh_every=args.teacher_refresh_every,
                                 name="lagged-self", **common)
    return SessionTeacher(client, name=source, **common)
